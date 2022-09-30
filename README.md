### Virtual environment

```bash
python -m venv venv
& venv/scripts/activate
python -m pip install pip setuptools wheel
python -m pip install -e .
```

## Install PyTorch
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

### Objective 
⭐Motivated by Andrej Karpathy's experiments with OpenAI Whisper.⭐ 
Get transcriptions of Lex Fridman episodes from [karpathy.ai](https://karpathy.ai/lexicap/).
Convert .vtt to text files. [parsing vtt, discarding timecodes, merging chronologically close lines into a larger block, and outputting
the result in a human-readable txt file. [Motivation](https://medium.com/@morga046/creating-an-nlp-data-set-from-youtube-subtitles-fb59c0955c2)]
Train GPT model on text files.