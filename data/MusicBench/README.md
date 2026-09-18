

---
license: cc-by-sa-3.0
---

# MusicBench Dataset

The MusicBench dataset is a music audio-text pair dataset that was designed for text-to-music generation purpose and released along with Mustango text-to-music model. MusicBench is based on the MusicCaps dataset, which it expands from 5,521 samples to 52,768 training and 400 test samples!


## Dataset Details
MusicBench expands MusicCaps by:
1. Including music features of chords, beats, tempo, and key that are extracted from the audio.
2. Describing these music features using text templates and thus enhancing the original text prompts.
3. Expanding the number of audio samples by performing musically meaningful augmentations: semitone pitch shifts, tempo changes, and volume changes.

Train set size = 52,768 samples
Test set size = 400



### Dataset Description
MusicBench consists of 3 .json files and attached audio files in .tar.gz form.

The train set contains audio augmented samples and enhanced captions. Additionally, it offers ChatGPT rephrased captions for all the audio samples.
Both TestA and TestB sets contain the same audio content, but TestB has all 4 possible control sentences (related to 4 music features) in captions of all samples, while TestA has no control sentences in the captions.

For more details, see Figure 1 in our paper.


Each row of a .json file has:
1. **location** (of the files after decompressing the .tar.gz file)
2. **main_caption** – text prompts that are a result of augmentation (TestB contains control sentences, train set contains ChatGPT rephrased captions here)
3. **alt_caption** – in the case of TestB these are captions without any control sentences added.
4. prompt_aug – A control sentence related to volume change augmentation.
5. prompt_ch – A control sentence describing the chord sequence.
6. prompt_bt – A control sentence describing the beat count (meter)
7. prompt_bpm – A control sentence describing tempo, either in beats per minute (bpm), or in musical words, e.g., Adagio, Moderato, Presto.
8. prompt_key – A control sentence related to the extracted musical key.
9. **beats** – The beat and downbeat timestamps. This is used as an input for training Mustango.
10. bpm – The tempo feature saved as a number.
11. **chords** – The chord sequence contained in the track. This is used as an input for training Mustango.
12. **chords_time** – Timestamps of the detected chords. This is used as an input for training Mustango.
13. key – The root and the type of the detected key.
14. keyprob – The confidence score for this detected key provided by the detection algorithm.
14. is_audioset_eval_mcaps – Whether this sample (in its non-augmented form) is a part of Audioset (and MusicCaps) eval (True) or train (False) set.

# FMACaps Evaluation Dataset
Hereby, we also present you the FMACaps evaluation dataset which consists of 1000 samples extracted from the Free Music Archive (FMA) and pseudocaptioned through extracting tags from audio and then utilizing ChatGPT in-context learning. More information is available in our paper!

Most of the samples are 10 second long, exceptions are between 5 to 10 seconds long.

Data size: 1,000 samples
Sampling rate: 16 kHz

Files included:
1. 1,000 audio files in the "audiodata" folder
2. FMACaps_A – this file contains captions with NO control sentences.
3. FMACaps_B – this file contains captions with ALL control sentences. We used this file the our controllability evaluation of Mustango.
4. FMACaps_C – this file contains captions with SOME controls sentences. For each sample, we chose 0/1/2/3/4 control sentences with a probability of 25/30/20/15/10 %, as described in our paper. This file was used to objectively evaluate audio quality of Mustango.

The structure of each .json file is identical to MusicBench, as described in the previous section, with the exception of "alt_caption" column being empty. **All captions** are in the **"main_caption" column**!

## Links

- **Code Repository:** [https://github.com/AMAAI-Lab/mustango]
- **Paper:** [https://arxiv.org/abs/2311.08355]
- **Demo:** [https://replicate.com/declare-lab/mustango]
- **Website:** [https://amaai-lab.github.io/mustango/]


## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

```bibtex
@inproceedings{melechovsky2024mustango,
  title={Mustango: Toward Controllable Text-to-Music Generation},
  author={Melechovsky, Jan and Guo, Zixun and Ghosal, Deepanway and Majumder, Navonil and Herremans, Dorien and Poria, Soujanya},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={8286--8309},
  year={2024}
}
```


**License:** cc-by-sa-3.0