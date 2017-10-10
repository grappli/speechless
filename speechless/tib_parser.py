__author__ = 'Steffen'

import csv
from xml.etree import ElementTree
import librosa
import subprocess
from shutil import copyfileobj

def get_segments(file):
    # change encoding to allow german character reading without error
    firstline = "<?xml version='1.0' encoding='ISO8859-1'?>"
    newfile = 'temp.xml'
    with open(file, 'r', encoding='utf8') as from_file:
        with open(newfile, 'w', encoding='utf8') as to_file:
            from_file.readline()
            to_file.write(firstline)
            copyfileobj(from_file, to_file)

    segments = []
    e = ElementTree.parse(newfile).getroot().findall('text')[0]
    for item in e.findall('phrase'):
        begin = int(item.get('start'))
        end = int(item.get('end'))
        phrase = item.text
        segments.append({'begin': begin, 'end': end, 'phrase': phrase})
    return segments

def cut_wav(wav_file, segments):
    data = librosa.load(wav_file, sr=16000)[0]
    files = []
    for segment in segments:
        wav_segment = data[segment['begin'] * 16000:segment['end'] * 16000]
        file = wav_file + '_' + str(segment['begin']) + '.wav'
        librosa.output.write_wav(file, wav_segment, sr=16000)
        files.append(file)

    return files

filename = '11406'

command = 'avconv  -i /data/TIB_dataset/videos/{}.mp4 -ac 1 -ar 16000 -vn /data/TIB_dataset/videos/{}.wav'.format(filename, filename)
subprocess.call(command, shell=True)

segs = get_segments('/data/TIB_dataset/transcripts/{}.xml'.format(filename))
files = cut_wav('/data/TIB_dataset/videos/{}.wav'.format(filename), segs)

with open('file.csv', 'w', encoding='utf8') as csvfile:
    writer = csv.writer(csvfile)
    for i, seg in enumerate(segs):
        writer.writerow(('{}_{}'.format(filename, seg['begin']), files[i], seg['phrase'], 'train'))
