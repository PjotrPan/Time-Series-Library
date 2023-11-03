# Time Series Library (TSlib)
TSlib is an open-source library for deep learning researchers, especially deep time series analysis.

We provide a neat code base to evaluate advanced deep time series models or develop your own model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**


## Information for HSE project

Bevor die Modelle trainiert werden können, muss der Datensatz entsprechend vorbereitet werden. Die Rohdaten liegen in XML vor und sollen später als Pandas DataFrame geladen werden. Dafür werden die Daten in einer csv zwischengespeichert. Dafür führt ihr einmal das data_provider/ohio.py Skript aus. Die notwendige Ordnerstruktur (dataset) muss manuell erstellt werden. 

```
├── checkpoints
├── data_provider
├── dataset
│   └── T1DMOhio
│       ├── 2020
│       ├── raw
│       ├── single
│       └── target
├── exp
├── layers
├── models
├── pic
├── results
├── scripts
├── test_results
├── utils
├── venv
└── wandb
```

Da für die Bewertung lediglich die Testdatensätze der Patienten aus 2020 verwendet werden, können die einzelnen Patientendaten verschieden kombiniert werden. Dafür haben wir 3 Modi vorgesehen:
 - Single: Verwendet nur die Daten eines einzigen Patienten für Training, Validierung und Test
 - Target: Verwendet die Daten (train und test) aller Patienten außer des 'Targets' für das Training und die Daten des 'Targets' für Validierung (train) und Test (test) 
 - 2020: Im Gegensatz zu den beiden vorherigen Modi, wird hier ein globales Modell trainiert. Alle Daten von Patienten aus 2018 werden für das Training verwendet, während die Daten der Patienten aus 2020 für Validierung und Test verwendet werden.

Folgende Attribute sind bereits implementiert und können als Argument für das Training übergeben werden: glucose_level, finger_stick, basal, temp_basal, bolus, basis_gsr, basis_air_temperature, basis_heart_rate, basis_skin_temperature, sleep, work, exercise, meal

Außerdem können insulin und activity als Attributnamen übergeben werden. Hintergrund ist die Idee, mehrere lichte Attribute in einem dichten Attribut zusammenzufassen. 
- insulin: Hier werden die Basal und Bolus Werte zusammengefasst. Temp_basal überschreibt die eigentliche Basalrate und Bolus wird bei Einnahme addiert.
- activity: Da sich Schlaf, Arbeit und Sport jeweils gegenseitig ausschließen, wird hier eine Aktivitätsskala erzeugt mit 0 für wenig aktiv und 1 für sehr aktiv.
Der MA-Filter in der Datensatzklasse glättet die Zeitreihe der forecast history, jedoch nicht die der ground truth.

Für die Interpolation nicht vorhandener Werte verwenden wir die [Standardmethoden](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) von Pandas. 

Im Folgenden werden die wichtigsten CLI Argumente beschrieben:

name | info | type
-- | - | -
seq_len | Anzahl der Zeitschritte, die für die Prädiktion betrachtet werden | int
label_len | Anzahl der Zeitschritte aus der forecast history, die als Hinweis/Richtung in die Prädiktion mitgegeben werden| int
pred_len | Anzahl der Zeitschritte, die vorhergesagt werden sollen | int 
patient_number | Nummer des Patienten, auf welchen der Fokus gelegt werden soll | int
sweep | Startet einen vordefinierten Hyperparameter-Sweep| bool
features | Namen der Features, die für das Training verwendet werden sollen | string
filter_size | Größe des Moving Average Filters ( MA-(filter_size*2 + 1)) | int
scaler | Scalertyp | string
interpolation_method | Methode mit welcher nan-Werte interpoliert werden sollen | str



## Leaderboard for Time Series Analysis

Till February 2023, the top three models for five different tasks are:

| Model<br>Ranking | Long-term<br>Forecasting                                     | Short-term<br>Forecasting                                    | Imputation                                                   | Anomaly<br>Detection                                         | Classification                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| 🥇 1st            | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)                 | [TimesNet](https://arxiv.org/abs/2210.02186)       |
| 🥈 2nd            | [DLinear](https://github.com/cure-lab/LTSF-Linear)           | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [Non-stationary<br/>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer) |
| 🥉 3rd            | [Non-stationary<br>Transformer](https://github.com/thuml/Nonstationary_Transformers) | [FEDformer](https://github.com/MAZiqing/FEDformer)           | [Autoformer](https://github.com/thuml/Autoformer)            | [Informer](https://github.com/zhouhaoyi/Informer2020)        | [Autoformer](https://github.com/thuml/Autoformer)  |

**Note: We will keep updating this leaderborad.** If you have proposed advanced and awesome models, welcome to send your paper/code link to us or raise a pull request. We will add them to this repo and update the leaderborad as soon as possible.

**Compared models of this leaderboard.** ☑ means that their codes have already been included in this repo.

  - [x] **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
  - [x] **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)
  - [x] **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py)
  - [x] **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/ETSformer.py)
  - [x] **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Nonstationary_Transformer.py)
  - [x] **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py)
  - [x] **Pyraformer** - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [[ICLR 2022]](https://openreview.net/pdf?id=0EXmFzUn5I) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Pyraformer.py)
  - [x] **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py)
  - [x] **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Informer.py)
  - [x] **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Reformer.py)
  - [x] **Transformer** - Attention is All You Need [[NeurIPS 2017]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)

See our latest paper [[TimesNet]](https://arxiv.org/abs/2210.02186) for the comprehensive benchmark. We will release a real-time updated online version soon.

**Newly added baselines.** We will add them into the leaderboard after a comprehensive evaluation.
  - [x] **FiLM** - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/forum?id=zTQdHSQUQWc)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py)
  - [x] **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol) [[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
  - [x] **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/MICN.py)
  - [x] **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)[[Code]](https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py)
 
## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtained the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/84fbc752d0e94980a610/) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy). Then place the downloaded data under the folder `./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate model. We provide the experiment scripts of all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```

## Contact
If you have any questions or suggestions, feel free to contact:

- Haixu Wu (whx20@mails.tsinghua.edu.cn)
- Tengge Hu (htg21@mails.tsinghua.edu.cn)
- Haoran Zhang (z-hr20@mails.tsinghua.edu.cn)

or describe it in Issues.

## Acknowledgement

This project is supported by the National Key R&D Program of China (2021YFB1715200).

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://github.com/thuml/Flowformer

All the experiment datasets are public and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer

- Classification: https://www.timeseriesclassification.com/
