from setuptools import setup, find_packages

setup(
    name='illama',
    version='0.1.0',
    description='A fast, lightweight, parallel inference server for Llama LLMs.',
    author='Nick Potafiy',
    author_email='nick@veridia.ai',
    url='https://github.com/nickpotafiy/illama',
    packages=find_packages(),
    install_requires=[
        'exllamav2',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
)