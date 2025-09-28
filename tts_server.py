import os
import sys
import tempfile
import zmq
import torch
import time

# ✅ Set HuggingFace mirror endpoint for China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Ensure online mode for runtime downloads
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

# Add IndexTTS to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "index-tts"))

from indextts.infer import IndexTTS

class IndexTtsServer:
    def __init__(self, model_size="base"):
        self.service_name = "text-to-wav"  # Keep same service name

        print("Loading IndexTTS model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = os.environ.get('DEVICE', device)
        print(f"IndexTTS device: {device}")
        
        # Model directory configuration
        model_dir = os.environ.get('INDEX_TTS_MODEL_DIR', os.path.join(current_dir, "checkpoints"))
        config_path = os.path.join(model_dir, "config.yaml")
        
        print(f"Using model directory: {model_dir}")
        print(f"Using config path: {config_path}")
        
        # Verify model files exist
        required_files = [
            "bigvgan_generator.pth",
            "bpe.model", 
            "gpt.pth",
            "config.yaml"
        ]
        
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                print(f"❌ Required file not found: {file_path}")
                raise FileNotFoundError(f"Missing required model file: {file}")
            else:
                print(f"✅ Found: {file_path}")

        # Load IndexTTS model
        try:
            self.tts = IndexTTS(
                cfg_path=config_path,
                model_dir=model_dir,
                is_fp16=False,  # reverting back to false
                device=device,
                use_cuda_kernel=False  # revert backed to false, got error for deepspeed
            )
            print("IndexTTS model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize IndexTTS: {e}")
            raise

        # Default voice file - can be overridden per request
        self.default_voice = os.environ.get('INDEX_TTS_DEFAULT_VOICE', 
                                          os.path.join(current_dir, "coXTTS.wav"))
        
        if os.path.exists(self.default_voice):
            print(f"✅ Default voice file found: {self.default_voice}")
        else:
            print(f"⚠️  Default voice file not found: {self.default_voice}")
            print("Voice file will need to be provided in each request")

        # ZMQ setup - keep same architecture
        self.url = os.environ.get('ZMQ_BACKEND_ROUTER_URL', 'tcp://localhost:5560')
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect(self.url)

        # Register with broker - same service name
        self.socket.send_multipart([self.service_name.encode()])

        # Warm up models to avoid first-request delays
        self.warm_up_models()

    def warm_up_models(self):
        """Warm up IndexTTS model with dummy text to avoid first-request delays"""
        print("Warming up IndexTTS model...")
        dummy_text = "测试"  # Simple Chinese text
        
        # Create temp file for dummy output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
            try:
                # Use default voice if available, otherwise skip warmup
                if os.path.exists(self.default_voice):
                    self.process_text(dummy_text, temp_file.name, self.default_voice)
                    print("IndexTTS model warmed up successfully!")
                else:
                    print("Skipping warmup - no default voice file available")
            except Exception as e:
                print(f"Warning: Failed to warm up models: {e}")

    def process_text(self, text_data, output_file, voice_file=None):
        print("Received text data for synthesis:")
        print("→", text_data)
        print("Saving audio to:", output_file)
        
        if voice_file:
            print("Using voice file:", voice_file)
        else:
            voice_file = self.default_voice
            print("Using default voice file:", voice_file)

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Use IndexTTS inference
            start_time = time.perf_counter()
            
            # Previous behavior:
            #   We called infer_fast directly in full FP32. Turning on is_fp16/use_cuda_kernel
            #   in this repo pulled in DeepSpeed and attempted to compile CUDA ops, which
            #   fails in our production image (no CUDA toolkit/CUDA_HOME).
            # Change:
            #   Use PyTorch AMP (FP16 autocast) + inference_mode to get T4 speedups without
            #   requiring DeepSpeed or a full CUDA toolchain. Ops that lack FP16 support
            #   fall back to FP32 automatically inside autocast.
            with torch.inference_mode():
                # Use the new torch.amp.autocast API (torch>=2.0). The old torch.cuda.amp.autocast is deprecated.
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    self.tts.infer_fast(
                        audio_prompt=voice_file,
                        text=text_data,
                        output_path=output_file,
                        verbose=False,
                        max_text_tokens_per_sentence=100,
                        sentences_bucket_max_size=4,
                        do_sample=False,  # for production T4 GPU
                        top_p=0.8,
                        top_k=30,
                        temperature=1.0,
                        length_penalty=0.0,
                        num_beams=1, # original num_beams is 3
                        repetition_penalty=10.0,
                        max_mel_tokens=600
                    )
            
            inference_time = time.perf_counter() - start_time
            print(f"✅ IndexTTS inference completed in {inference_time:.2f} seconds")

            if not os.path.exists(output_file):
                print("❌ Audio file was not created!")
                return "Error: IndexTTS failed to generate audio."
            else:
                file_size = os.path.getsize(output_file)
                print(f"✅ Audio file created successfully. Size: {file_size} bytes")
                return "Text processed successfully!"
                
        except Exception as e:
            print(f"❌ Exception during IndexTTS processing: {e}")
            return f"Error in text processing: {str(e)}"

    def run(self):
        while True:
            try:
                print("IndexTTS server is ready to receive a message ...")
                message = self.socket.recv_multipart()
                payload = message[3]

                temp_dir = tempfile.mkdtemp()
                temp_audio_path = os.path.join(temp_dir, "response.wav")

                # For now, use default voice file
                # In a more advanced implementation, you could parse the payload
                # to extract voice file information if needed
                status_msg = self.process_text(payload.decode('utf-8'), temp_audio_path)
                print("Status:", status_msg)
                # add more logs and test
                if os.path.exists(temp_audio_path):
                    with open(temp_audio_path, 'rb') as audio_file:
                        audio_data = audio_file.read()
                else:
                    audio_data = b""

                self.socket.send_multipart(message[:3] + [audio_data])

                try:
                    os.remove(temp_audio_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp files: {str(e)}")

            except Exception as e:
                error_msg = f"Error processing request: {str(e)}"
                print(error_msg)
                self.socket.send_multipart(message[:3] + [b""])

def main():
    sys.stdout = sys.stderr
    server = IndexTtsServer()
    server.run()

if __name__ == "__main__":
    main()
