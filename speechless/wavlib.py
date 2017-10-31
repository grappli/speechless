__author__ = 'Steffen'

import requests
import re
import numpy as np
import librosa
import os, random
import sys

class WavTools:
    def __init__(self, directory):
        self.directory = str(directory) + '/wav'

    @staticmethod
    def download_from_tag(tag='field-recording', directory='environmental', num_files=100):
        urls = WavTools.scrape_sound_urls(tag, num_files)
        WavTools.download_files(urls, directory)
        WavTools.segment_wavs(directory)

    @staticmethod
    def scrape_sound_urls(tag='field-recording', num_files=100):
        urls = list()
        pages = num_files / 15 + 1
        for page_num in range(int(pages)):
            url = 'https://freesound.org/search/?q={}&f=license%3A%22Creative+Commons+0%22+type%3Awav&page={}'.format(tag, page_num + 1)
            page = requests.get(url).text
            regex = re.compile('href="(/people/(.*?)/sounds/(\d*?)/)"')
            for match in regex.findall(page):
                urls.append(match[0])
        return urls[:num_files]

    @staticmethod
    def segment_wavs(directory):
        files = WavTools.absoluteFilePaths(directory)
        for file in files:
            wav_data, _ = librosa.load(str(file), sr=16000, res_type='kaiser_fast')
            # segment into 10-second chunks
            for i in range(0, len(wav_data), 160000):
                stop_index = len(wav_data) if stop_index > len(wav_data) else i + 160000
                data = wav_data[i:stop_index]
                output_wav_file = os.path.splitext(file)[0] + str(int(i / 1000)) + '.wav'
                librosa.output.write(output_wav_file, data, 16000)
            os.remove(file)

    @staticmethod
    def download_files(urls, directory='wavfiles'):

        s = requests.Session()
        loginpage = s.get('https://freesound.org/home/login/?next=/')
        token = re.findall("name='csrfmiddlewaretoken' value='(.*?)'", loginpage.text)[0]
        postdata = 'csrfmiddlewaretoken={}&username=grappli&password=7d63lPFcTiT2&next=%2F'.format(token)

        s.post('https://freesound.org/home/login/', data=postdata,
                              headers={'Referer': 'https://freesound.org/home/login/?next=/',
                                       'HTTP_X_CSRFTOKEN': token,
                                       'Content-Type': 'application/x-www-form-urlencoded'})
        for url in urls:
            soundpage = s.get('http://freesound.org' + url)
            download_url = url + 'download/'
            filename = re.findall(download_url + '(.*?)"', soundpage.text)[0]
            response = s.get('http://freesound.org' + download_url)
            with open('{}/{}'.format(directory, filename), 'wb') as outfile:
                outfile.write(response.content)

    @staticmethod
    def mix_wav(foreground_wav, foreground_volume=0.6, background_volume=0.4):
        desired_sample_rate = 16000
        background_data = np.array()

        # load foreground
        foreground_data, foreground_rate = librosa.load(str(foreground_wav),
                                                        sr=desired_sample_rate, res_type='kaiser_fast')

        # load enough background wavs to cover length of foreground
        first = True

        while len(background_data) < len(foreground_data):

            background_wav = WavTools.get_next_wav()
            wav_background_data, wav_background_rate = librosa.load(str(background_wav),
                                                        sr=desired_sample_rate, res_type='kaiser_fast')
            if first:
                first = False
                background_data = background_data[WavTools.curr_wav_start:]

            background_data = np.concatenate(background_data, wav_background_data, axis=0)

        # shorten background to be same length as foreground
        WavTools.curr_wav_start = len(background_data) - len(foreground_data)
        WavTools.curr_wav_idx -= 1
        background_data = background_data[:len(foreground_data)]

        # Generate output
        output = foreground_data * foreground_volume + \
                 background_data[:len(foreground_data)] * background_volume

        return output

    @staticmethod
    def mix_wavs_raw(foreground_wav, background_wav, foreground_volume=0.6, background_volume=0.4):
        desired_sample_rate = 16000

        foreground_data, foreground_rate = librosa.load(str(foreground_wav),
                                                        sr=desired_sample_rate, res_type='kaiser_fast')
        background_data, background_rate = librosa.load(str(background_wav),
                                                        sr=desired_sample_rate, res_type='kaiser_fast')
        
        foreground_length = 1.0 * len(foreground_data) / foreground_rate
        background_length = 1.0 * len(background_data) / background_rate

        # Downsample both to 16 kHz - not necessary because of using librosa but keep just in case
        # get_interp = lambda x, length: np.arange(0, length, 1.0 / x)
        # foreground_data = np.array([np.interp(get_interp(desired_sample_rate, foreground_length),
        #                             get_interp(foreground_rate, foreground_length),
        #                             foreground_data[:,i]) for i in range(2)]).transpose()
        # background_data = np.array([np.interp(get_interp(desired_sample_rate, background_length),
        #                             get_interp(background_rate, background_length),
        #                             background_data[:,i]) for i in range(2)]).transpose()

        # loop background if necessary
        if foreground_length > background_length:
            loops = int(len(foreground_data) * 1.0 / len(background_data)) + 1
            background_data = np.repeat(background_data, loops, axis=0)
        
        # Generate output
        output = foreground_data * foreground_volume + \
                 background_data[:len(foreground_data)] * background_volume

        return output

    @staticmethod
    def absoluteFilePaths(directory):
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))

    @staticmethod
    def get_background_wav_list(wavdir):
        WavTools.wavlist = WavTools.absoluteFilePaths(wavdir)

    wavlist = []
    curr_wav_idx = 0
    curr_wav_start = 0

    @staticmethod
    def get_next_wav():
        numwavs = len(WavTools.wavlist)
        if numwavs > 0:
            curr_wav = WavTools.wavlist[WavTools.curr_wav_idx]
            WavTools.curr_wav_idx += 1
            if numwavs <= WavTools.curr_wav_idx:
                WavTools.curr_wav_idx = 0
            return curr_wav
        return ''

    @staticmethod
    def mix_wavs_to_file(foreground_wav, background_wav, output_wav_file,
                 foreground_volume=0.6, background_volume=0.4, sample_rate=16000):

        data = WavTools.mix_wavs_raw(foreground_wav, background_wav, foreground_volume, background_volume)
        librosa.output.write(output_wav_file, data, sample_rate)

    def random_wav(self, directory):
        wavdir = self.directory + '/' + directory
        return wavdir + '/' + random.choice(os.listdir(wavdir))

if __name__ == "__main__":
    directory = sys.argv[1]
    WavTools.download_from_tag(directory='/data/speechless-data/wav/environmental')