__author__ = 'Steffen'

import csv
from xml.etree import ElementTree
import librosa
import subprocess

def get_segments(file):
    segments = []
    e = ElementTree.parse(file).getroot().findall('text')[0]
    for item in e.findall('phrase'):
        begin = item.get('start')
        end = item.get('end')
        phrase = item.text
        segments.append({'begin': begin, 'end': end, 'phrase': phrase})
    return segments

def cut_wav(wav_file, segments):
    data = librosa.load(wav_file + '.wav', sr=16000)[0]
    files = []
    for segment in segments:
        wav_segment = data[segment['begin'] * 16000:segment['end'] * 16000]
        file = wav_file + '_' + str(segment['begin']) + '.wav'
        librosa.write_wav(file, wav_segment, sr=16000)
        files.append(file)

    return files

filename = '11406'

command = 'avconv  -i /data/videos/{}.mp4 -ac 1 -ar 16000 -vn /data/videos/{}.wav'.format(filename, filename)
subprocess.call(command, shell=True)

segs = get_segments('/data/transcripts/{}.xml'.format(filename))
files = cut_wav('/data/videos/{}.wav'.format(filename), segs)

with open('file.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i, seg in enumerate(segs):
        writer.writerow(('{}_{}'.format(filename, seg['begin']), files[i], seg['phrase'], 'train'))
