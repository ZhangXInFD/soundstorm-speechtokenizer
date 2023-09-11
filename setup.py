from pathlib import Path
from setuptools import setup

NAME = 'soundstorm_speechtokenizer'
DESCRIPTION = 'SoundStorm built upon SpeechTokenizer'
URL = 'https://github.com/ZhangXInFD/soundstorm-speechtokenizer'
EMAIL = 'xin_zhang22@m.fudan.edu.cn'
AUTHOR = 'Xin Zhang'
REQUIRES_PYTHON = '>=3.8.0'

for line in open('soundstorm_speechtokenizer/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']
        
HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
    
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['soundstorm_speechtokenizer'],
    install_requires=['speechtokenizer',
                      'accelerate',
                      'beartype',
                      'lion-pytorch'],
    include_package_data=True,
    license='MIT License',
    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
    ])