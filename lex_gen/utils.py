import webvtt
import os
import pandas as pd
import numpy as np
from glob import glob

# -----------------------------------------------------------------------------
def convert_vtt(filenames):
    """ Convert vtt files to text files """
    for filename in filenames:
        webvtt.from_srt(filename).save_as_srt(filename.replace('.vtt', '.srt'))
    # create asset folder if one doesn't already exist
    if os.path.isdir('{}/text'.format(os.getcwd())) == False:
        os.makedirs('text')
    # extract the text and times from the vtt file
    for file in filenames:
        captions = webvtt.read(file)
        text_time = pd.DataFrame()
        text_time['text'] = [caption.text for caption in captions]
        text_time['start'] = [caption.start for caption in captions]
        text_time['stop'] = [caption.end for caption in captions]
        text_time.to_csv('transcripts/text/{}.csv'.format(file[18:-4]), index=False) # -4 to remove '.vtt'
        # remove files from local drive
        os.remove(file)


def prepare_corpus(inPath: str, outPath: str) -> None:
    # collect paths for each transcript file
    paths = glob(f"{inPath}*.csv", recursive=True)

    # extract all text from each individual file and append it to list
    texts = []
    for file in paths:
        df = pd.read_csv(file)
        text = "".join(df.text)
        texts.append(text)

    with open(f"{outPath}.corpus.txt", "w", encoding="utf-8") as f:
        f.write(str(texts))