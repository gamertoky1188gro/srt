import os
import whisper1

# Set your folder path here
folder_path = r"C:/Users/tokyi/PycharmProjects/PythonProject/cv-corpus-21.0-delta-2025-03-14-en/cv-corpus-21.0-delta-2025-03-14/en/clips"

count = 0
max_files = 10

whisper1.add_task("C:/Users/tokyi/PycharmProjects/PythonProject/downloads/I Built a Roller Coaster In My Lab.mp3", job="txt, txt with file name, twfn, srt, vtt, json, csv, html, md, lines, docx, pdf, latex, tsv, xml, yaml, ass, sbv, dfxp, ttml, cue, chapters, jsonl, stats, frequency, rtf, xlsx, json_srt, vxml, edi, audio_aligned_json, eaf, textgrid, webembed", tran = "fra_Latn,spa_Latn,deu_Latn,ita_Latn,rus_Cyrl,cmn_Hans,ara_Arab,jpn_Jpan,kor_Hang,hin_Deva")  # Example URL
whisper1.add_task("C:/Users/tokyi/PycharmProjects/PythonProject/downloads/I Built a Roller Coaster In My Lab.mp3", job="txt, txt with file name, twfn, srt, vtt, json, csv, html, md, lines, docx, pdf, latex, tsv, xml, yaml, ass, sbv, dfxp, ttml, cue, chapters, jsonl, stats, frequency, rtf, xlsx, json_srt, vxml, edi, audio_aligned_json, eaf, textgrid, webembed")  # Example URL

# Execute all tasks at once
results = whisper1.execute_task()
for r in results:
    print(f"{r['file']}: {r['text']}")
