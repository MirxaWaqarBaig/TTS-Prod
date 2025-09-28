# TTS-Prod - IndexTTS Production System

A production-ready Text-to-Speech (TTS) system based on IndexTTS-1.5, featuring zero-shot voice cloning, multi-language support (Chinese/English), and microservices architecture for scalable deployment.

[![GitHub](https://img.shields.io/badge/GitHub-TTS--Prod-blue)](https://github.com/MirxaWaqarBaig/TTS-Prod)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://github.com/MirxaWaqarBaig/TTS-Prod)
[![IndexTTS](https://img.shields.io/badge/IndexTTS-1.5-green)](https://github.com/index-tts/index-tts)

## 🚀 Features

- **Zero-shot Voice Cloning**: Clone any voice from 3-10 seconds of reference audio
- **Multi-language Support**: Chinese and English text-to-speech
- **High-Quality Audio**: State-of-the-art audio quality with BigVGAN2 vocoder
- **Production Ready**: Docker-based microservices with ZMQ messaging
- **GPU Acceleration**: CUDA support with automatic CPU fallback
- **Scalable Architecture**: Message queue-based distributed processing

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional, CPU fallback available)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 10GB+ free space for models and dependencies

### Software Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Container Toolkit** (for GPU support)
- **Git**: For cloning repositories

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MirxaWaqarBaig/TTS-Prod.git
cd TTS-Prod
```

### 2. Clone IndexTTS Source Code
```bash
# Clone IndexTTS from its official repository
git clone https://github.com/index-tts/index-tts.git

# Verify the IndexTTS directory structure
ls -la index-tts/

# Expected structure should include:
# - indextts/ (main package)
# - setup.py
# - requirements.txt
# - README.md
# - tests/
# - tools/
```

### 3. Download Model Checkpoints
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download IndexTTS-1.5 model files using huggingface-cli
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml \
  bigvgan_discriminator.pth \
  bigvgan_generator.pth \
  bpe.model \
  dvae.pth \
  gpt.pth \
  unigram_12000.vocab \
  --local-dir checkpoints

# Alternative: Download using wget (if huggingface-cli is not available)
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P checkpoints
wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P checkpoints

# Verify all model files are downloaded
ls -la checkpoints/

# Expected files and approximate sizes:
# - config.yaml (~2KB) - Model configuration
# - bigvgan_discriminator.pth (~1.2GB) - BigVGAN discriminator
# - bigvgan_generator.pth (~1.2GB) - BigVGAN generator
# - bpe.model (~2MB) - Byte-pair encoding model
# - dvae.pth (~200MB) - Discrete VAE model
# - gpt.pth (~2.5GB) - GPT model weights
# - unigram_12000.vocab (~1MB) - Vocabulary file
# Total size: ~5GB
```

**Project Structure After Setup:**
```
TTS-Prod/
├── index-tts/              # IndexTTS source code (cloned)
├── checkpoints/            # Model files
├── tts_server.py          # Main TTS server
├── docker-compose_TTS.yaml # Docker orchestration
├── Dockerfile.TTS         # TTS server Dockerfile
├── requirements_TTS.txt   # Python dependencies
├── test_tts.sh           # Test script
├── coXTTS.wav            # Default voice file
└── README.md             # This documentation
```


### 4. Verify GPU Support (Optional)
```bash
# Test NVIDIA Docker support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## 🐳 Docker Build & Deployment

### 1. Build Docker Images

#### Build TTS Server Image
```bash
# Build the IndexTTS server image
docker build -f Dockerfile.TTS -t ss-tts .
```

#### Build Additional Service Images
You'll need to build the broker and gateway images from their respective repositories:

```bash
# Build broker and gateway images from their repositories
# (These services are available in separate repositories)
docker build -t ss-broker <broker-repo-url>
docker build -t ss-gateway <gateway-repo-url>
```

### 2. Deploy with Docker Compose

```bash
# Start all services
docker-compose -f docker-compose_TTS.yaml up -d

# Check service status
docker-compose -f docker-compose_TTS.yaml ps

# View logs
docker-compose -f docker-compose_TTS.yaml logs -f
```

### 3. Verify Deployment
```bash
# Check if all containers are running
docker ps

# Test TTS service
curl -X POST http://localhost:8000/api/text-to-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the TTS system."}' \
  --output test_output.wav
```

## 🧪 Testing

### 1. Run Test Script
```bash
# Make the test script executable
chmod +x test_tts.sh

# Run the test
./test_tts.sh
```

### 2. Manual API Testing

#### Local Testing
```bash
# Test with Chinese text
curl -X POST http://localhost:8000/api/text-to-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "大家好，这是中文语音合成测试。"}' \
  --output chinese_test.wav

# Test with English text
curl -X POST http://localhost:8000/api/text-to-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is an English TTS test."}' \
  --output english_test.wav
```

#### Production Testing
```bash
# Test with production endpoint
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "安排验车**：商家会指派验车人员，并与您协商确定具体的验车时间和地点。"}' \
  --output production_test.wav
```

### 3. Voice Cloning Test

#### Local Testing
```bash
# Test with custom voice (replace with your audio file)
curl -X POST http://localhost:8000/api/text-to-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a voice cloning test.", "voice_file": "path/to/your/voice.wav"}' \
  --output cloned_voice.wav
```

#### Production Testing
```bash
# Test voice cloning with production endpoint
curl -X POST http://chatbot.sharestyleai.com:8000/api/text-to-wav \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a production voice cloning test.", "voice_file": "path/to/your/voice.wav"}' \
  --output production_cloned_voice.wav
```

## ⚙️ Configuration

### Environment Variables

#### TTS Server Configuration
```bash
# Device selection (cuda/cpu)
export DEVICE=cuda

# Model directory
export INDEX_TTS_MODEL_DIR=/app/checkpoints

# Default voice file
export INDEX_TTS_DEFAULT_VOICE=/app/coXTTS.wav

# ZMQ broker URL
export ZMQ_BACKEND_ROUTER_URL=tcp://broker:5560
```

#### Docker Compose Configuration
Edit `docker-compose_TTS.yaml` to customize:
- Port mappings
- Resource limits
- Environment variables
- Network configuration

### Model Configuration
The model configuration is in `checkpoints/config.yaml`. Key parameters:
- **Sample Rate**: 24000 Hz
- **Model Dimensions**: 1280
- **Layers**: 24 transformer layers
- **Attention Heads**: 20
- **Max Text Tokens**: 600
- **Max Mel Tokens**: 800

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size or use CPU
export DEVICE=cpu
```

#### 2. Model Loading Errors
```bash
# Check if all model files exist
ls -la checkpoints/
# Should contain: bigvgan_generator.pth, gpt.pth, dvae.pth, bpe.model, config.yaml
```

#### 3. ZMQ Connection Issues
```bash
# Check if broker is running
docker logs sharestyle-broker

# Restart services
docker-compose -f docker-compose_TTS.yaml restart
```

#### 4. Audio Quality Issues
- Ensure reference audio is clear and 3-10 seconds long
- Use 24kHz sample rate for best results
- Check if GPU is being utilized properly

### Performance Optimization

#### GPU Optimization
```bash
# Enable mixed precision
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda
```

#### Memory Optimization
```bash
# Reduce model precision
# Edit tts_server.py and set is_fp16=True
```

## 📊 Monitoring & Logs

### View Service Logs
```bash
# All services
docker-compose -f docker-compose_TTS.yaml logs

# Specific service
docker-compose -f docker-compose_TTS.yaml logs tts-server

# Follow logs in real-time
docker-compose -f docker-compose_TTS.yaml logs -f tts-server
```

### Health Checks
```bash
# Check service health
curl http://localhost:8000/health

# Check GPU usage
nvidia-smi

# Check memory usage
docker stats
```

## 🚀 Production Deployment

### 1. Production Configuration
```yaml
# docker-compose.prod.yaml
version: '3.8'
services:
  tts-server:
    image: ss-tts
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Load Balancing
```bash
# Scale TTS service
docker-compose -f docker-compose_TTS.yaml up --scale tts-server=3
```

## 📈 Performance Benchmarks

### Hardware Requirements
- **Minimum**: 4 CPU cores, 8GB RAM, CPU-only
- **Recommended**: 8 CPU cores, 16GB RAM, NVIDIA T4 GPU
- **Optimal**: 16 CPU cores, 32GB RAM, NVIDIA V100/A100 GPU

### Performance Metrics
- **Inference Time**: ~2-5 seconds per sentence (GPU)
- **Memory Usage**: ~4-8GB RAM
- **Concurrent Requests**: 2-4 per GPU
- **Audio Quality**: 24kHz, 16-bit PCM

## 🤝 Contributing

1. Fork the [TTS-Prod repository](https://github.com/MirxaWaqarBaig/TTS-Prod)
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support & Issues

- **Repository**: [TTS-Prod on GitHub](https://github.com/MirxaWaqarBaig/TTS-Prod)
- **Issues**: [Report issues here](https://github.com/MirxaWaqarBaig/TTS-Prod/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MirxaWaqarBaig/TTS-Prod/discussions)

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) - Core TTS model
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) - Vocoder
- [XTTS](https://github.com/coqui-ai/TTS) - Base architecture
- [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) - Inspiration

## 🔗 Quick Links

- **GitHub Repository**: [https://github.com/MirxaWaqarBaig/TTS-Prod](https://github.com/MirxaWaqarBaig/TTS-Prod)
- **IndexTTS Original**: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
- **Docker Hub**: [Available on GitHub Packages](https://github.com/MirxaWaqarBaig/TTS-Prod/packages)

For issues and questions:
- Create an issue in the [TTS-Prod repository](https://github.com/MirxaWaqarBaig/TTS-Prod/issues)
- Check the troubleshooting section
- Review the logs for error details

---

**Note**: This system requires significant computational resources. For production use, ensure adequate GPU memory and processing power for optimal performance.
