## AIセキュリティの情報まとめ

## AIの脆弱性
### 2020年
 * (Last revised 2 Apr 2020) [Real-world adversarial attack on MTCNN face detection system](https://arxiv.org/abs/1910.06261)  
 実環境における誤認識誘発攻撃。  
 医療用フェイスマスクや頬等に特殊な柄のパッチを貼り付けることで、MTCNNベースの顔検出器による顔検知を回避する手法。デジタルドメインのみならず、実環境でも利用可能。  
 検証コード：[Real-world attack on MTCNN face detection system](https://github.com/edosedgar/mtcnnattack)  

* (Submitted on 1 Apr 2020) [Evading Deepfake-Image Detectors with White- and Black-Box Attacks](https://arxiv.org/abs/2004.00622)  
 ホワイトボックス/ブラックボックス設定の誤認識誘発攻撃。  
 DeepFakes検知器を攻撃する手法。1pixelの反転、全体の1%に摂動を加える等、ホワイトボックスとブラックボックス設定の攻撃手法を提案している。  

* (Submitted on 31 Mar 2020) [Adversarial Attacks on Multivariate Time Series](https://arxiv.org/abs/2004.00410)  
 TBA.  

* (Submitted on 31 Mar 2020) [Inverting Gradients -- How easy is it to break privacy in federated learning?](https://arxiv.org/abs/2003.14053)  
 勾配ベースのデータ窃取攻撃。  
 Federated learningにおいて、勾配を共有する仕組みが安全ではない事を示した研究。共有される勾配から高解像度の入力画像を復元し、DNNのプライバシー侵害が可能であることを実証している。  

 * (Submitted on 30 Mar 2020) [DeepHammer: Depleting the Intelligence of Deep Neural Networks through Targeted Chain of Bit Flips](https://arxiv.org/abs/2003.13746)  
 ハードウェアレベルの誤認識誘発攻撃。  
 Row hammerの脆弱性を利用し、DNNモデルの重みビットを反転させることで、DNNモデルの推論精度を低下させる。  

 * (Submitted on 28 Mar 2020) [DaST: Data-free Substitute Training for Adversarial Attacks](https://arxiv.org/abs/2003.12703)  
 ブラックボックス設定の誤認識誘発攻撃。  
 既存のブラックボックス誤認識誘発攻撃は、攻撃対象モデルの学習データに近似したデータを用いて（Adversarial Examplesを作成するための）代替モデルを作成しているが、本手法は生成モデル（GAN）で生成したデータを用いるとのこと。本手法で作成したAdversarial ExamplesをMicrosoft Azure上の機械学習モデルに分類させたところ、攻撃成功率は98.35%であったとのこと。実データを使用しないブラックボックス誤認識誘発攻撃はあまり例がない。  
 関連研究：[Adversarial Imitation Attack](https://arxiv.org/abs/2003.12760)  

 * (Submitted on 28 Mar 2020) [Policy Teaching via Environment Poisoning: Training-time Adversarial Attacks against Reinforcement Learning](https://arxiv.org/abs/2003.12909)  
 強化学習（以下、RL）に対する報酬汚染攻撃。  

 * (Submitted on 27 Mar 2020) [Adaptive Reward-Poisoning Attacks against Reinforcement Learning](https://arxiv.org/abs/2003.12613)  
 強化学習（以下、RL）に対する報酬汚染攻撃。  
 各学習のエポックにおいて、報酬`rt`に細工を加えた汚染報酬`rt+δt`を作成し、RLエージェントに悪意のあるPolicyを学習させる。  

### 2019年
 * (Last revised 20 Aug 2019) [Targeted Adversarial Examples for Black Box Audio Systems](https://arxiv.org/abs/1805.07820)  
 音声の誤認識誘発攻撃。  
 オリジナル音声にノイズを入れて音声アシスタントを騙す手法。遺伝的アルゴリズムし、効率よくAdversarial exampleを生成。  
 検証コード: [rtaori/Black-Box-Audio](https://github.com/rtaori/Black-Box-Audio)  

 * (Llast revised 1 Jul 2019) [GenAttack: Practical Black-box Attacks with Gradient-Free Optimization](https://arxiv.org/abs/1805.11090)  
 誤認識誘発攻撃。  
 遺伝的アルゴリズムを使用し、効率良くAdversarial exampleを生成。少ないクエリ数で攻撃可能。  

 * (Last revised 28 Jun 2019) [Black-box Adversarial Attacks on Video Recognition Models](https://arxiv.org/abs/1904.05181)  
 リアルタイム物体認識に対する誤認識誘発攻撃。  
 ブラックボックス設定で攻撃が可能。  

 * (Last revised 8 Jun 2019) [Adversarial camera stickers: A Physical Camera Attack on Deep Learning Classifier](https://arxiv.org/abs/1904.00759)  
 実環境における誤認識誘発攻撃。  
 カメラのレンズに特殊な模様を付けたステッカーを貼り付けることで、物体認識モデルの判断を誤らせる手法。  
 デモ動画：[Adversarial Camera Sticker fooling ResNet-50 model](https://www.youtube.com/watch?v=wUVmL33Fx54)  

 * (Submitted on 16 May 2019) [Data Poisoning Attacks on Stochastic Bandits](https://arxiv.org/abs/1905.06494)  
 強化学習（以下、RL）に対する報酬汚染攻撃。  

 * (Submitted on 18 Apr 2019) [Fooling automated surveillance cameras: adversarial patches to attack person detection](https://arxiv.org/abs/1904.08653)  
 リアルタイム物体認識に対する誤認識誘発攻撃。  
 特殊な柄のパッチを人間が身に着けることで、物体認識器から人間を秘匿する手法。  
 検証コード: [adversarial-yolo](https://gitlab.com/EAVISE/adversarial-yolo)  

 * (Submitted on 8 Apr 2019) [Adversarial Audio: A New Information Hiding Method and Backdoor for DNN-based Speech Recognition Models](https://arxiv.org/abs/1904.03829)  
 音声の誤認識誘発攻撃。  
 オリジナル音声に人間には聞こえない別音声を埋め込む。細工音声を訓練した音声認識システムは埋め込み音声を認識できるが、他の音声認識システムは認識できない。バックドア攻撃にも転用できる可能性がある。  

 * (Last revised 4 Apr 2019) [Discrete Attacks and Submodular Optimization with Applications to Text Classification
](https://arxiv.org/abs/1812.00151)  
テキスト分類器に対する誤認識誘発攻撃。  

 * (Submitted on 27 Mar 2019) [Rallying Adversarial Techniques against Deep Learning for Network Security
](https://arxiv.org/abs/1903.11688)  
侵入検知システムに対する誤認識誘発攻撃。  

### 2018年
 * (Last revised 10 Nov 2018) [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)  
 データ汚染攻撃。  
 学習データに少量の汚染データを紛れ込ませることで、決定境界を意図した方向に操作する手法。Webやメールボックスから定期的にデータを収集して再学習するモデルを標的にしている。  
 検証コード: [ashafahi/inceptionv3-transferLearn-poison](https://github.com/ashafahi/inceptionv3-transferLearn-poison)  

 * (Last revised 30 Oct 2018) [Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding](https://arxiv.org/abs/1808.05665)  
 音声の誤認識誘発攻撃。  
 音声に高周波数のノイズを加えることで、音声アシスタントを騙す手法。  

 * (Last revised 24 Aug 2018) [Data Poisoning Attacks in Contextual Bandits](https://arxiv.org/abs/1808.05760)  
 強化学習（以下、RL）の報酬汚染攻撃。  
 RLエージェントの報酬を細工することで、攻撃者が意図した行動をRLエージェントに学習させる手法。  

 * (Submitted on 18 Jul 2018) [Gradient Band-based Adversarial Training for Generalized Attack Immunity of A3C Path Finding](https://arxiv.org/abs/1807.06752)  
 強化学習（以下、RL）の環境汚染攻撃。  
 RLエージェントが参照する地図データを細工することで、エージェントの経路選択を誤らせる手法。  

 * (Last revised 11 Jul 2018) [Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/abs/1804.08598)  
 ブラックボックス設定の誤認識誘発攻撃。  
 少ないクエリアクセスでAdversarial Examplesを作成し、Google Cloud Vision APIを騙すことが可能。  
 検証コード: [labsix/limited-blackbox-attacks](https://github.com/labsix/limited-blackbox-attacks/blob/master/attacks.py)  

 * (Last revised 7 Jun 2018) [Synthesizing Robust Adversarial Examples](https://arxiv.org/abs/1707.07397)  
 実環境における誤認識誘発攻撃。  
 ノイズ・歪み・アフィン変換に頑健性のある2次元のAdversarial Examples画像を複雑な3次元物体に適用し、リアルタイムカメラを騙すAdversarial Examplesを作成。  

 * (Last revised 10 Apr 2018) [Robust Physical-World Attacks on Deep Learning Models](https://arxiv.org/abs/1707.08945)  
 実環境における誤認識誘発攻撃。  
 自動運転自動車を念頭に、実環境にAdversarial exampleを適用した研究。道路標識に細工を加えることで、画像認識システムの判断を誤らせる手法。  
 参考情報: [標識にシールを貼って自動運転カーを混乱に陥れるハッキング技術「Robust Physical Perturbations(RP2)」](https://gigazine.net/news/20170807-robust-physical-perturbations/)  

 * (Submitted on 1 Apr 2018) [Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning](https://arxiv.org/abs/1804.00308)  
 線形回帰モデルに対するデータ汚染攻撃。  
 学習データに少量の汚染データを加えることで、モデルの予測結果を意図した方向に操作する。  

### 2017年
 * (Submitted on 23 May 2017) [Black-Box Attacks against RNN based Malware Detection Algorithms](https://arxiv.org/abs/1705.08131v1)  
 アンチマルウエアに対する誤認識誘発攻撃。  
 RNNベースのアンチマルウエアを騙す手法。  

### 2016年
 * (Submitted on 19 Dec 2016) [Simple Black-Box Adversarial Perturbations for Deep Networks](https://arxiv.org/abs/1612.06299)  
 誤認識誘発攻撃。  
 Convolutional Neural Network（以下、CNN）への入力画像を微小に細工することでCNNの判断を誤らせる。画素数が小さい画像では1pixel、大きな画像でも数pixelの細工で誤分類させることが可能。  
 検証ブログ: [Convolutional Neural Networkに対する攻撃手法 -誤分類の誘発-](https://www.mbsd.jp/blog/20170516.html)  

### 未分類
 * [Membership Inference Attacks against Machine Learning Models](https://arxiv.org/abs/1610.05820)  
 攻撃対象モデルの学習データを取得する手法。  
 攻撃対象モデルの入力・出力から独自のモデルを作成し、学習済みのデータと未学習のデータの使用して攻撃対象モデルの挙動を確認する。本論文では、GoogleやAmazonなどの商用サービスで作成されたモデルを使用して本手法を評価している。  

 * [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)  
 MLモデルへの入出力をブラックボックスアクセスで観測し、MLモデルの内部ロジックを復元する手法。  
 2項ロジスティック回帰や多項ロジスティック回帰、多層パーセプトロン、決定木等、多数のMLモデルを標的にしている。  
 検証コードは[こちら](https://github.com/ftramer/Steal-ML)  
 一部検証済み: [機械学習モデルに対する攻撃手法 -Equation-Solving Attacks-](https://www.mbsd.jp/blog/20170117.html)  

 * [Machine Learning with Membership Privacy using Adversarial Regularization](https://arxiv.org/abs/1807.05852)  
 TBA.  

 * [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)  
 学習時に使用される勾配を基に学習データを復元する手法。  
 マルチノードで分散学習するMLにおいては、MLモデル間で勾配を共有する事がある。そのようなケースを想定し、他MLから共有された勾配から学習データを復元する。  

 * [Neural Trojans](https://arxiv.org/abs/1710.00942v1)  
 TBA.  

 * [Hardware Trojan Attacks on Neural Networks](https://arxiv.org/abs/1806.05768)  
 TBA.  

 * [DeepLocker - Concealing Targeted Attacks with AI Locksmithing](https://www.blackhat.com/us-18/briefings/schedule/index.html#deeplocker---concealing-targeted-attacks-with-ai-locksmithing-11549)  
 Deep Neural Network（以下、DNN）を利用した標的型マルウエア。  
 平時はビデオアプリ等として動作し、（顔画像や音声等で）標的を認識するとランサムウエア化する。  
 ランサムウエアのペイロードを暗号化（ロック）しておき、DNNで標的を認識するとその復号鍵を生成して動作するため、既存のアンチマルウエアでは検知が難しいとの事。  

 * [Programmable Neural Network Trojan for Pre-Trained Feature Extractor](https://arxiv.org/abs/1901.07766)  
 TBA  
 
 * [Trojaning Attack on Neural Networks](https://github.com/PurduePAML/TrojanNN/blob/master/trojan_nn.pdf)  
 トリガとなるデータを既存モデルに入力することで、モデルに意図した出力を行わせる手法。  
 既存モデルの訓練データにアクセスする必要は無く、既存モデルをリバースエンジニアして作成した**Torojan Trigger**と**再訓練用の学習データ**を組み合わせることで、トリガデータを効率良く作成可能。攻撃の成功率も非常に高い。  
 検証コード：[PurduePAML/TrojanNN](https://github.com/PurduePAML/TrojanNN)  
 
 * [PoTrojan: powerful neural-level trojan designs in deep learning models](https://arxiv.org/abs/1802.03043)  
 ネットワークの隠れ層に**Triggerノード**と**Payloadノード**を挿入し、モデルに意図した出力を行わせる手法。  
 殆どの入力データをモデルは正しく分類する事が可能だが、ある特定のデータに対してのみ（攻撃者が意図した）誤った出力をするため、ステルス性が高い。  

 * [Backdooring Convolutional Neural Networks via Targeted Weight Perturbations](https://arxiv.org/abs/1812.03128)  
 CNNにTrojanを仕込む手法。  

 * [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)  
 TBA  
 
 * [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/abs/1811.03728)  
 TBA  
 
 * [TrojDRL: Trojan Attacks on Deep Reinforcement Learning Agents](https://arxiv.org/abs/1903.06638)  
 TBA  

 * [A backdoor attack against LSTM-based text classification systems](https://arxiv.org/abs/1905.12457)  
 LSTMベースの文書分類器にTrojanを仕込む方法。  

 * [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a530/19skfH8dcqc)  
 スライド：[Neural Cleanse](https://www.ieee-security.org/TC/SP2019/SP19-Slides-pdfs/Bolun_Wang_-MAC_-_08-Bolun_Wang-Neural_Clense_Identifying_and_Mitigating_Backdoor_Attacks_against_Neural_Networks_(1).pdf)  
 
 * [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets](https://arxiv.org/abs/1905.05897)  
 正しくラベル付けされたポイズニング画像が注入された学習データで学習を行うことで、画像認識モデルにバックドアを仕込む手法。  
 対象モデルの出力、ネットワーク構造、学習データにアクセスせずに攻撃を行うことが可能。  

 * [Summoning Demons: The Pursuit of Exploitable Bugs in Machine Learning](https://arxiv.org/abs/1701.04739)  
 OpenCVやScikit-Learn等の脆弱性を利用してDoSやBoF等を引き起こす手法。  

 * [Security Risks in Deep Learning Implementations](https://arxiv.org/abs/1711.11008)  
 TensorFlowやCaffe等が利用しているライブラリの脆弱性を利用してDoSやBoF等を引き起こす手法。  

 * [StuxNNet:Practical Live Memory Attacks on Machine Learning Systems](https://aivillage.org/material/cn18-norwitz/slides.pdf)  
 実行中のNeural Network（以下、NN）の判断を誤らせる手法。  
 NN実行中のメモリに展開されたNNの重みやバイアスを直接操作し、NNに誤判断を引き起こさせる。  
 検証コード：[https://github.com/bryankim96/stux-DNN](bryankim96/stux-DNN)  

 * [Practical Fault Attack on Deep Neural Networks](https://arxiv.org/abs/1806.05859)  
 組み込み機器にビルドインされたDNNの判断を誤らせる手法。  
 機器にレーザを照射し、DNNの隠れ層の計算結果に誤りを生じさせる。  

## AIの脆弱性診断ツール
 * [CleverHans](https://github.com/tensorflow/cleverhans)  
 誤認識誘発攻撃に対する耐性を評価するツール。  

 * [Foolbox](https://github.com/bethgelab/foolbox)  
 誤認識誘発攻撃に対する耐性を評価するツール。  

 * [AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for Attacking Black-box Neural Networks](https://arxiv.org/abs/1805.11770)  
 誤認識誘発攻撃に対する耐性を評価するツール。  

 * [Adversarial Robustness Toolbox for Machine Learning Models](https://www.blackhat.com/us-18/arsenal/schedule/index.html#adversarial-robustness-toolbox-for-machine-learning-models---arsenal-theater-demo-12026)  
 誤認識誘発攻撃に対する耐性を評価するツール。  

 * [MLSploit: Resilient ML Platform - Advanced Deep Learning Analytic Platform Made Easy for Every Security Researcher](https://www.blackhat.com/us-18/arsenal/schedule/index.html#mlsploit-resilient-ml-platform---advanced-deep-learning-analytic-platform-made-easy-for-every-security-researcher-11798)  
 誤認識誘発攻撃に対する耐性を評価するツール。  
 検証コード: [intel/Resilient-ML-Research-Platform](https://github.com/intel/Resilient-ML-Research-Platform)  

 * [ADAGIO: Interactive Experimentation with Adversarial Attack and Defense for Audio](https://arxiv.org/abs/1805.11852)  
 音声の誤認識誘発攻撃に対する耐性を評価するツール。  

 * [Deep Pwning](https://github.com/cchio/deep-pwning)  
 機械学習版のMetasploit。  
 更新は止まっている。  
 検証コード: [cchio/deep-pwning](https://github.com/cchio/deep-pwning)  

 * [DeepSec](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a398/19skfzVqzGE)  
  誤認識誘発攻撃に対する耐性を評価するツール。  

 * [Comprehensive Privacy Analysis of Deep Learning](https://www.computer.org/csdl/proceedings-article/sp/2019/666000b021/19skg8ZskUM)  
 AIからの情報漏えいをチェックする手法。  
 参考情報: [Comprehensive Privacy Analysis of Deep Learning](https://www.ieee-security.org/TC/SP2019/SP19-Slides-pdfs/Milad_Nasr_-_08-Milad_Nasr-Comprehensive_Privacy_Analysis_of_Deep_Learning_(1).pdf)  

## 参考情報
 * [ADVERSARIAL MACHINE LEARNING TUTORIAL](https://aaai18adversarial.github.io/)  
 Adversarial攻撃のチュートリアル。防御手法も掲載されている。  
 情報提供：[@icoxfog417](https://twitter.com/icoxfog417) 氏  
 
以上
