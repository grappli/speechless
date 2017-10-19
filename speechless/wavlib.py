__author__ = 'Steffen'

import requests
import re
import numpy as np
import librosa
import os, random

class WavTools:
    def __init__(self, directory):
        self.directory = str(directory) + '/wav'

    @staticmethod
    def download_from_tag(tag='field-recording', directory='environmental', num_files=100):
        urls = WavTools.scrape_sound_urls(tag, num_files)
        WavTools.download_files(urls, directory)

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
    def mix_wavs_to_file(foreground_wav, background_wav, output_wav_file,
                 foreground_volume=0.6, background_volume=0.4, sample_rate=16000):

        data = WavTools.mix_wavs_raw(foreground_wav, background_wav, foreground_volume, background_volume)
        librosa.output.write(output_wav_file, data, sample_rate)

    def random_wav(self, directory):
        wavdir = self.directory + '/' + directory
        return wavdir + '/' + random.choice(os.listdir(wavdir))
