__author__ = 'Steffen'

import requests
import re
import numpy as np
from scikits.audiolab import wavread, wavwrite
import os, random

def download_from_tag(tag='field-recording', directory='environmental', num_files=100):
    urls = scrape_sound_urls(tag, num_files)
    download_files(urls, directory)

def scrape_sound_urls(tag='field-recording', num_files=100):
    urls = list()
    pages = num_files / 15 + 1
    for page_num in range(pages):
        url = 'https://freesound.org/search/?q={}&f=license%3A%22Creative+Commons+0%22+type%3Awav&page={}'.format(tag, page_num + 1)
        page = requests.get(url).text
        regex = re.compile('href="(/people/(.*?)/sounds/(\d*?)/)"')
        for match in regex.findall(page):
            urls.append(match[0])
    return urls[:num_files]

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

def mix_wavs_raw(foreground_wav, background_wav, foreground_volume=0.6, background_volume=0.4):
    foreground_data, foreground_rate, fg_enc  = wavread(foreground_wav)
    background_data, background_rate, bg_enc = wavread(background_wav)

    # Make sure both are stereo
    if len(foreground_data.shape) != 2:
        foreground_data = np.repeat(foreground_data[:,np.newaxis], 2, axis=1)
    if len(background_data.shape) != 2:
        background_data = np.repeat(background_data[:,np.newaxis], 2, axis=1)

    # Downsample both to 16 kHz
    desired_sample_rate = 16000
    foreground_length = 1.0 * foreground_data.shape[0] / foreground_rate
    background_length = 1.0 * background_data.shape[0] / background_rate
    get_interp = lambda x, length: np.arange(0, length, 1.0 / x)
    foreground_data = np.array([np.interp(get_interp(desired_sample_rate, foreground_length),
                                get_interp(foreground_rate, foreground_length),
                                foreground_data[:,i]) for i in range(2)]).transpose()
    background_data = np.array([np.interp(get_interp(desired_sample_rate, background_length),
                                get_interp(background_rate, background_length),
                                background_data[:,i]) for i in range(2)]).transpose()

    # loop background if necessary
    if foreground_length > background_length:
        loops = int(len(foreground_data) * 1.0 / len(background_data)) + 1
        background_data = np.repeat(background_data, loops, axis=0)

    # Generate output
    output = foreground_data * foreground_volume + \
             background_data[:len(foreground_data), :] * background_volume

    return output

def mix_wavs_to_file(foreground_wav, background_wav, output_wav_file,
             foreground_volume=0.6, background_volume=0.4):

    data = mix_wavs_raw(foreground_wav, background_wav, foreground_volume, background_volume)
    wavwrite(data, output_wav_file, desired_sample_rate)

def random_wav(directory='wavfiles'):
    return directory + random.choice(os.listdir(directory))