# Whisper Transcription & Translation Toolkit
This project is a comprehensive Python toolkit for transcribing and translating audio and video files using OpenAI's Whisper models and Meta's NLLB-200 translation model. It supports batch processing, a wide variety of output formats, and can handle both local files and YouTube/online media URLs.

## Contact

For questions, support, or to report issues, please visit the [GitHub Issues page](https://github.com/gamertoky1188gro/srt/issues).

## Table of Contents

- [Features](#features)
- [Upcoming Features](#upcoming-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Run the Script](#1-run-the-script)
  - [Add Tasks (Programmatic Usage)](#2-add-tasks-programmatic-usage)
  - [Output Files](#3-output-files)
  - [Supported Output Formats](#4-supported-output-formats)
  - [Translation](#5-translation)
  - [Command-Line Example](#6-command-line-example)
- [Configuration](#configuration)
- [Model Storage](#model-storage)
- [Troubleshooting](#troubleshooting)
- [Extending](#extending)
- [Demo](#demo)
- [Known Issues & Limitations](#known-issues--limitations)
- [License](#license)
- [Credits](#credits)
- [FAQ](#faq)
- [Performance Tips](#performance-tips)

## Features

- **Transcription**: Uses Whisper models (tiny, base, small, medium, large, turbo, large-v2, large-v3) for high-quality speech-to-text.
- **Translation**: Supports translation of transcripts and subtitle segments into 200+ languages using Meta's NLLB-200 model.
- **Batch Processing**: Queue multiple files or URLs for batch transcription and translation.
- **Flexible Input**: Accepts local audio files or URLs (YouTube, playlists, etc.).
- **Automatic Download**: Downloads audio/video from URLs using `yt-dlp`.
- **Rich Output Formats**: Generates transcripts in TXT, SRT, VTT, CSV, JSON, HTML, Markdown, DOCX, PDF, LaTeX, TSV, XML, YAML, ASS, SBV, DFXP/TTML, CUE, chapters, JSONL, stats, frequency CSV, RTF, XLSX, JSON SRT, VoiceXML, EDI, audio-aligned JSON, ELAN EAF, Praat TextGrid, and web-embeddable HTML.
- **Segmented & Full Text**: Outputs both full transcript and per-segment (subtitle) files.
- **Translation Per Segment**: Subtitle formats (SRT, VTT, etc.) can be generated in translated languages.
- **GPU Acceleration**: Automatically uses CUDA if available for both Whisper and translation models.
- **Configurable Model Storage**: Prompts for a folder to store models on first run.
- **Progress Bars**: Uses `tqdm` for progress indication if installed.

## Upcoming Features

The following features are planned for future releases:

- **Web UI or GUI**: Build a simple web interface (using Flask, FastAPI, or Streamlit) or a desktop GUI for easier use by non-technical users.
- **Speaker Diarization**: Integrate speaker identification to label “who spoke when” in transcripts.
- **Language Auto-Detection**: Automatically detect the spoken language in audio files before transcription.
- **Audio Preprocessing**: Add noise reduction, silence trimming, or volume normalization before transcription.
- **Cloud Storage Integration**: Support for uploading/downloading files from Google Drive, Dropbox, or S3.
- **Job Scheduling & Monitoring**: Add a queue system with job status tracking and notifications (email, webhook, etc.).
- **REST API**: Expose transcription and translation as a RESTful API for integration with other apps.
- **Custom Vocabulary/Boosting**: Allow users to provide custom words or phrases to improve recognition accuracy.
- **Batch Download from Playlists**: Enhanced playlist handling with metadata and progress reporting.
- **Post-Processing Tools**: Add grammar correction, summarization, or keyword extraction to transcripts.
- **Multi-Channel Audio Support**: Handle stereo/multi-channel audio and transcribe each channel separately.
- **User Profiles & History**: Save user preferences, history of jobs, and allow re-download of past results.
- **Dockerization**: Provide a Docker image for easy deployment.
- **Mobile App Companion**: Create a simple mobile app to upload audio and receive transcripts.
- **Advanced Analytics**: Visualize transcript statistics, word clouds, or speaker timelines.

## Requirements

- Python Python 3.13.5
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (for URL downloads)
- [openai-whisper](https://github.com/openai/whisper)
- [transformers](https://github.com/huggingface/transformers) and [sentencepiece](https://github.com/google/sentencepiece) (for translation)
- [torch](https://pytorch.org/) (for model inference)
- Optional: `tqdm`, `python-docx`, `fpdf`, `openpyxl`, `pyyaml` for extra output formats

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/gamertoky1188gro/srt.git
   cd <repo>
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Install yt-dlp:**
   - On Windows:
     ```sh
     pip install yt-dlp
     ```
   - Or download the binary from [yt-dlp releases](https://github.com/yt-dlp/yt-dlp#installation).

## Usage

### 1. Run the Script

```sh
python whisper1.py
```
- On first run, you will be prompted to select a folder to store Whisper and translation models.
- You will be prompted to select a Whisper model (e.g., `large-v3`).

### 2. Add Tasks (Programmatic Usage)

You can use the `add_task` function to queue files or URLs for transcription/translation:

```python
from whisper1 import add_task, execute_task

# Add a local file for English transcript and SRT
add_task('audio.mp3', 'txt,srt')

# Add a YouTube URL for transcript, SRT, and Spanish translation
add_task('https://www.youtube.com/watch?v=xxxx', 'txt,srt', 'spa_Latn')

# Run all tasks
execute_task()
```

### 3. Output Files

- For each input file, a folder is created in the same directory as the audio file, named after the file (without extension).
- All output formats are saved in this folder, e.g.:
  - `audio/audio_all_transcripts.txt`
  - `audio/audio.srt`
  - `audio/audio_spa_Latn.srt` (Spanish translation)
  - `audio/audio.json`, `audio/audio.csv`, etc.

### 4. Supported Output Formats

- `txt`, `txt with file name`, `twfn`, `srt`, `vtt`, `csv`, `json`, `html`, `md`, `lines`, `docx`, `pdf`, `latex`, `tsv`, `xml`, `yaml`, `ass`, `sbv`, `dfxp`, `ttml`, `cue`, `chapters`, `jsonl`, `stats`, `frequency`, `rtf`, `xlsx`, `json_srt`, `vxml`, `edi`, `audio_aligned_json`, `eaf`, `textgrid`, `webembed`
- See code for details on each format.

### 5. Translation

- Specify translation target languages as a comma-separated list of NLLB-200 language codes (e.g., `fra_Latn,spa_Latn`).
- Both full transcript and per-segment translations are supported.

### 6. Command-Line Example

You can modify the `if __name__ == "__main__":` block to add your own logic for batch jobs, or import and use the functions in your own scripts.

## Configuration

This project uses a configuration file and user prompts to manage model storage and settings:

### Configuration File
- **Path:** The configuration file is stored at `~/.whisper_config.json` in your home directory.
- **Purpose:** It stores the path to the folder where Whisper and translation models are downloaded and cached.
- **Format:** JSON. Example:
  ```json
  {
    "model_dir": "C:/Users/yourname/whisper"
  }
  ```
- **How it's set:**
  - On first run, you are prompted to select a folder for model storage via a file dialog.
  - The selected folder is saved in the config file for future runs.

### Model Storage
- All Whisper and translation models are stored in the folder specified by `model_dir` in the config file, under a `whisper` subfolder.
- You can change the model storage location by deleting or editing the config file; the script will prompt you again on next run.

### Environment Variables
- This project does not require any special environment variables by default.
- If you want to override the config location or other settings, you can set environment variables in your shell before running the script (advanced users only).

### Customization
- You can manually edit `~/.whisper_config.json` to change the model directory.
- For advanced configuration (e.g., CUDA/CPU selection), modify the code in `whisper1.py` as needed.

## Model Storage

- Models are stored in a user-selected folder (prompted on first run), under a `whisper` subfolder.
- The config is saved in your home directory as `.whisper_config.json`.

## Troubleshooting

- **Missing dependencies**: The script will prompt you to install missing packages for certain output formats.
- **CUDA not available**: The script will fall back to CPU if no GPU is detected.
- **yt-dlp not found**: Ensure `yt-dlp` is installed and in your PATH.
- **Model download issues**: Check your internet connection and available disk space.

## Extending

- Add new output formats by extending the `output_jobs` function in `execute_task()`.
- Add new translation languages by specifying the correct NLLB-200 language codes.

## Demo

A full demo of the toolkit's output files and capabilities is available in the following folder:

[Demo Output Folder](https://github.com/gamertoky1188gro/srt/tree/main/downloads/I%20Built%20a%20Roller%20Coaster%20In%20My%20Lab)

This demo includes:
- Example audio and video files
- All supported transcript and subtitle output formats
- Translations and segment-level outputs
- Web-embeddable HTML, JSON, and more

Explore the folder to see real outputs generated by this project.

## Known Issues & Limitations

1. **Output File Location**: Output files may sometimes be created in unexpected directories, especially if the input file paths are unusual or if the script is run from different working directories. Users should check the output directory printed in the terminal.
2. **Translation Reliability**: The translation feature may not always work as expected, especially if required dependencies (`transformers`, `sentencepiece`, or model files) are missing or if the translation model fails to load. Some translation errors are caught and reported in the output files.
3. **Dependency Requirements**: Some output formats (e.g., DOCX, PDF, XLSX, YAML) require extra Python packages. If these are not installed, the script will skip those outputs and print a warning.
4. **Large File/Batch Processing**: Processing very large files or many files at once can consume significant memory and may cause the script to crash or slow down, especially on systems with limited resources.
5. **No Speaker Diarization**: The script does not currently support identifying different speakers in the audio.
6. **No Language Auto-Detection**: The language of the audio must be known or set; there is no automatic language detection.
7. **No Cloud Integration**: All files must be local; there is no built-in support for cloud storage or remote files (except for YouTube/online downloads).
8. **No Web/GUI**: The script is command-line only; there is no graphical or web interface.
9. **Error Handling**: Some errors (e.g., file not found, permission denied, model download failure) may not be handled gracefully and could cause the script to exit.
10. **Translation Model Download**: The first use of translation may be slow due to model download, and failures in this step may not be clearly reported.

## License

This project is provided as-is for research and educational purposes. See individual model licenses for Whisper and NLLB-200.

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Meta NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

For questions or contributions, please open an issue or pull request.

## FAQ

1. **What is this project for?**
   - It transcribes and translates audio/video files using Whisper and NLLB-200 models.
2. **Which models are supported?**
   - All OpenAI Whisper models and Meta's NLLB-200 for translation.
3. **How do I install the requirements?**
   - Use `pip install -r requirements.txt` and install `yt-dlp`.
4. **How do I run the script?**
   - Run `python whisper1.py` in your terminal.
5. **How do I select a model?**
   - The script prompts you to select a model on first run.
6. **Where are models stored?**
   - In the folder you select on first run, saved in `~/.whisper_config.json`.
7. **How do I change the model storage location?**
   - Delete or edit `~/.whisper_config.json` and rerun the script.
8. **Can I process YouTube or online URLs?**
   - Yes, just pass the URL to `add_task()`.
9. **What output formats are supported?**
   - TXT, SRT, VTT, CSV, JSON, HTML, DOCX, PDF, and many more.
10. **How do I translate transcripts?**
    - Pass target language codes (e.g., `spa_Latn`) to `add_task()`.
11. **Can I batch process multiple files?**
    - Yes, queue as many tasks as you want before calling `execute_task()`.
12. **How do I get segment-level subtitles?**
    - Use output jobs like `srt`, `vtt`, etc.
13. **How do I get a full transcript?**
    - Use the `txt` output job.
14. **How do I get both original and translated outputs?**
    - Specify both jobs and translation languages in `add_task()`.
15. **What if a required package is missing?**
    - The script will prompt you to install it.
16. **Does it use GPU?**
    - Yes, if CUDA is available; otherwise, it uses CPU.
17. **How do I force CPU usage?**
    - Modify the code in `get_whisper_model()` and `get_translation_pipeline()`.
18. **Where are output files saved?**
    - In a folder named after the audio file, in the same directory as the input.
19. **Why are some output files missing?**
    - Check for errors in the terminal; some formats require extra packages.
20. **How do I add new output formats?**
    - Extend the `output_jobs` function in `whisper1.py`.
21. **How do I add new translation languages?**
    - Use the correct NLLB-200 language codes.
22. **Can I use this as a library in my own scripts?**
    - Yes, import `add_task` and `execute_task` from `whisper1.py`.
23. **How do I get help with errors?**
    - Check the Troubleshooting section or open an issue on GitHub.
24. **What if yt-dlp is not found?**
    - Ensure it is installed and in your PATH.
25. **How do I update the models?**
    - Delete the model files and rerun the script to re-download.
26. **Can I use this on Mac/Linux?**
    - Yes, but GUI prompts may differ; all core features work cross-platform.
27. **How do I contribute?**
    - See the Contributing section or open a pull request.
28. **Is there a web or GUI version?**
    - Not yet, but it is planned (see Upcoming Features).
29. **Why is translation not working?**
    - There are known issues; fixes are planned for future updates.
30. **How do I reset all settings?**
    - Delete `~/.whisper_config.json` and any downloaded models.
31. **Can I use my own Whisper model weights?**
    - Yes, place your model file in the model directory and select it.
32. **How do I specify multiple output formats?**
    - Use a comma-separated list in the `job` argument (e.g., `txt,srt,csv`).
33. **How do I specify multiple translation languages?**
    - Use a comma-separated list in the `tran` argument (e.g., `spa_Latn,fra_Latn`).
34. **What if the script crashes during processing?**
    - Check the error message and ensure all dependencies are installed.
35. **How do I get word-level timestamps?**
    - Use output jobs like `audio_aligned_json`.
36. **Can I generate subtitles for video files?**
    - Yes, as long as the video contains audio.
37. **How do I process a playlist?**
    - Pass the playlist URL to `add_task()`; all items will be processed.
38. **How do I get statistics about the transcript?**
    - Use the `stats` output job.
39. **How do I get word frequency data?**
    - Use the `frequency` output job.
40. **How do I get a Markdown or HTML transcript?**
    - Use the `md` or `html` output job.
41. **How do I get a DOCX or PDF transcript?**
    - Use the `docx` or `pdf` output job (requires extra packages).
42. **How do I get a LaTeX transcript?**
    - Use the `latex` output job.
43. **How do I get a YAML or XML transcript?**
    - Use the `yaml` or `xml` output job.
44. **How do I get a TSV or CSV transcript?**
    - Use the `tsv` or `csv` output job.
45. **How do I get a TextGrid or EAF file?**
    - Use the `textgrid` or `eaf` output job.
46. **How do I get a web-embeddable transcript?**
    - Use the `webembed` output job.
47. **How do I get a JSONL or JSON SRT file?**
    - Use the `jsonl` or `json_srt` output job.
48. **How do I get a CUE or chapters file?**
    - Use the `cue` or `chapters` output job.
49. **How do I get an RTF transcript?**
    - Use the `rtf` output job.
50. **How do I get an EDI or VoiceXML file?**
    - Use the `edi` or `vxml` output job.
51. **How do I get DFXP/TTML or SBV/ASS subtitles?**
    - Use the `dfxp`, `ttml`, `sbv`, or `ass` output jobs.
52. **How do I get help with language codes?**
    - See the NLLB-200 documentation for supported codes.
53. **Can I use this for commercial purposes?**
    - Check the licenses for Whisper, NLLB-200, and this project.
54. **How do I report a bug or request a feature?**
    - Open an issue or pull request on GitHub.
55. **How do I see a demo of the outputs?**
    - See the Demo section in this README.
56. **How do I check where my output files are created?**
    - Check the output directory printed in the terminal after processing.
57. **What if my output files are not where I expect?**
    - See Known Issues & Limitations; check all subfolders and output logs.
58. **How do I reset the configuration?**
    - Delete `~/.whisper_config.json` and restart the script.
59. **How do I use this with cloud storage?**
    - Cloud integration is planned for future releases.
60. **How do I get notified when processing is done?**
    - Job notifications are planned for future releases.

## Performance Tips

- **Use a GPU (CUDA):** For large files or batch jobs, ensure you have a CUDA-capable GPU and the correct PyTorch version installed. The script will use GPU automatically if available.
- **Batch Processing:** Queue multiple files with `add_task()` before calling `execute_task()` to minimize model loading overhead.
- **Choose the Right Model:** Use smaller models (e.g., `tiny`, `base`) for faster but less accurate results; use larger models (`large-v3`) for best accuracy but slower speed.
- **Limit Output Formats:** Only request the output formats you need to reduce file writing time.
- **Close Other Applications:** Free up system resources by closing unnecessary programs, especially when running large models on CPU.
- **Use SSD Storage:** Store audio files and models on an SSD for faster read/write speeds.
- **Preprocess Audio:** Trim silence and reduce noise in audio files before transcription for better speed and accuracy.
- **Batch Translation:** Translate in batches (as implemented) for better GPU utilization.
- **Monitor Memory Usage:** Large models and batch jobs can use significant RAM; monitor usage to avoid crashes.
- **Update Dependencies:** Use the latest versions of `torch`, `transformers`, and `whisper` for performance improvements.
- **Use Command Line:** Run the script from the command line for best performance, rather than from within an IDE.
- **Avoid Large Playlists at Once:** For very large YouTube playlists, process in smaller batches to avoid timeouts or memory issues.
- **Check Output Directory:** Writing to network drives or slow disks can bottleneck performance; use local fast storage.
- **Disable Progress Bars:** If running in a non-interactive environment, disable `tqdm` for slightly faster execution.
