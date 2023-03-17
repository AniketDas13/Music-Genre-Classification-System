# Music Genre Classification System
A classification system designed to predict the genre of any musical track using CNN
-------------------------------------------------------------------------------------

The purpose of this project is to predict the genres of songs by using machine learning techniques. The system aims to find out the corresponding artist recommending more of his or her songs. For this purpose, feature extraction is done by using signal processing techniques, then machine learning algorithms are applied with those features to do a multiclass classification for music genres. This project targets a high market demand owing to its high accuracy score and shorter segmentation period for musical tracks.

The project pertains to a multi-classification problem and a recommendation approach that we intend to design applying multiple Machine Learning algorithms, also including classes of Neural Networks. Feature engineering comprises data pre-processing, data wrangling, and data visualization. Training and Testing a machine using the CNN Algorithm. Audio processing in which MFCC or Mel-Spectrogram is employed for Music Information Retrieval.

![image](https://user-images.githubusercontent.com/37765408/225859659-562f7186-97d1-446a-83a9-e3efb221ef03.png)

The dataset used in the project has been provided for reference:
- GTZAN Dataset - Music Genre Classification
- https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?select=genres_original
- A collection of 10 genres with 100 audio files each, all having a length of 30 seconds
- 70/30 train/test split

Exclusions: We have only considered .wav music files as our input. Any other file format is excluded from the input range.

Assumptions:
- The length of the music file has been assumed to be at least 30 seconds for audio segmentation of data.
- Only one musical file format is considered (.wav).
- Only a set of predefined genres have been considered for our project.

Aniket Das (AniketDas13)

Other Team Members:
Anurag Ganguly (Gangulys-99)
Anupam Chakrborty (AnupamChaks0101)
Ananya Paul (ananyapaul2021)

N.B.: The model may not achieve the best accuracy, though fiddling with the various parameters may enhance it.
