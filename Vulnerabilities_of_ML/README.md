## Vulnerabilities of Machine Learning
脆弱性のカテゴリ（仮）。  
内容は仮のものであり、現時点では正確性を欠く可能性があります。  

 * Adversarial example  
 * Boundary manipulation  
 * Stealing model and privacy data  
 * Trojan  
 * Unsafety framework and library  
 * Other  

### Adversarial example
 * [Simple Black-Box Adversarial Perturbations for Deep Networks](https://arxiv.org/abs/1612.06299)  
 画像のadversarial example。  
 Convolutional Neural Network（以下、CNN）への入力画像を微小に細工することで、CNNの判断を誤らせる手法。画素数の小さな画像では1pixel、大きな画像でも数pixelの細工で誤分類する。  

 * [Robust Physical-World Attacks on Deep Learning Models](https://arxiv.org/abs/1707.08945)  
 画像のadversarial example。  
 現実世界にadversarial exampleを適用した研究。  
 検証では、道路標識に細工を加えることで画像認識システムが判断を誤ったことを示している。  

 * [GenAttack: Practical Black-box Attacks with Gradient-Free Optimization](https://arxiv.org/abs/1805.11090)  
 画像のadversarial example。  
 最新の画像認識システムを騙すことが可能なadversarial exampleの作成手法。  
 遺伝的アルゴリズムでadversarial exampleを生成する。  
 既存の手法と比べると、数桁オーダの少ないクエリ数で攻撃可能とのこと。  

 * [Targeted Adversarial Examples for Black Box Audio Systems](https://arxiv.org/abs/1805.07820)  
 音声のadversarial example。  
 音声に微小なノイズを入れて音声認識システムの判断を誤らせる手法。  
 論文ではMozilla Deep Speechを標的にしている。ソースコードは[こちら](https://github.com/rtaori/Black-Box-Audio)。  

 * [Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding](https://arxiv.org/abs/1808.05665)  
 音声のadversarial example。  
 元の音声データに対し、人間が知覚できる閾値以下の細工を加えることで、音声認識システムの判断を誤らせる手法。  

 * [Black-Box Attacks against RNN based Malware Detection Algorithms](https://arxiv.org/abs/1705.08131v1)  
 マルウエアのadversarial example。  
 RNNベースのアンチマルウエアを誤認識させるadversarial exampleを生成する手法。  

 * [Gradient Band-based Adversarial Training for Generalized Attack Immunity of A3C Path Finding](https://arxiv.org/abs/1807.06752)  
 強化学習のadversarial example。  
 最適経路選択を行う強化学習エージェントの判断を誤らせる。エージェントに入力される地図に細工を加えることで、エージェントの判断に狂いを生じさせる。  

 * [Data Poisoning Attacks in Contextual Bandits](https://arxiv.org/abs/1808.05760)  
 強化学習のadversarial example。  
 バンディット問題を対象にした攻撃手法。報酬に摂動を加えることで、攻撃者が意図したアームを引かせることができる。  

### Boundary manipulation  
 * [Making & Breaking Machine Learning Anomaly Detectors in Real Life](https://www.slideshare.net/codeblue_jp/making-breaking-machine-learning-anomaly-detectors-in-real-life-by-clarence-chio-code-blue-2015)  
 訓練データに少数の細工データを紛れ込ませることで決定境界をpivotingさせる攻撃手法。  
 CODE BLUE 2015でClarence Chio氏により発表された手法。  

 * [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)  
 訓練データに少数の細工データを紛れ込ませることで決定境界をpivotingさせる攻撃手法。  
 定期的に学習データを収集して学習を重ねる転移学習を標的にしている。  

 * [Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning](https://arxiv.org/abs/1804.00308)  
 線形回帰モデルに対するポイゾニング手法。  
 訓練データに細工を加えることでMLモデルの予測結果を操作する。  

### Stealing model and privacy data
 * [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)  
 MLモデルへの入力と出力をブラックボックスアクセスで観測し、MLモデルの内部ロジックを抽出する攻撃手法。  
 LRやMLP,決定木などのモデルの抽出や、訓練データを復元する手法まで多岐に渡る。  

 * [Machine Learning with Membership Privacy using Adversarial Regularization](https://arxiv.org/abs/1807.05852)  
 MLモデルから学習データをブラックボックスアクセスで盗む手法（inference attack）の対策。  

### Trojan
 * [Neural Trojans](https://arxiv.org/abs/1710.00942v1)  
 機械学習版トロイの木馬。   
 ベンダが配布するMLモデルが細工されていることを前提とした攻撃手法。  

 * [Hardware Trojan Attacks on Neural Networks](https://arxiv.org/abs/1806.05768)  
 機械学習版トロイの木馬。  

 * [DeepLocker - Concealing Targeted Attacks with AI Locksmithing](https://www.blackhat.com/us-18/briefings/schedule/index.html#deeplocker---concealing-targeted-attacks-with-ai-locksmithing-11549)  
 DNNを使用した標的型マルウエア。  
 平時はビデオアプリとして振る舞い、ターゲットを認識するとランサムウエア化する。  
 ランサムウエアのペイロードを暗号化（ロック）しておき、DNNでターゲットを認識すると復号鍵を生成する。  

### Unsafety framework and library
 * [Summoning Demons: The Pursuit of Exploitable Bugs in Machine Learning](https://arxiv.org/abs/1701.04739)  
 機械学習を実装するフレームワークのバグを利用した攻撃手法。  
 OpenCVやScikit-Learnなどのバグを突いてDoSやBoFなどを引き起こす。  

 * [Security Risks in Deep Learning Implementations](https://arxiv.org/abs/1711.11008)  
 機械学習を実装するフレームワークのバグを利用した攻撃手法。  
 TensorFlowやCaffeなどが内部で利用しているライブラリのバグを突いてDoSやBoFなどを引き起こす。  

 * [Practical Fault Attack on Deep Neural Networks](https://arxiv.org/abs/1806.05859)  
 物理的な攻撃。  
 組み込み機器にビルドインされたDNNに誤分類を引き起こさせる手法。  
 機器にレーザを照射し、隠れ層の計算結果に誤りを生じさせる。  

### Other
 * [StuxNNet:Practical Live Memory Attacks on Machine Learning Systems](https://aivillage.org/material/cn18-norwitz/slides.pdf)  
 実行中のニューラルネットワークに誤判断を引き起こさせる手法。  
 メモリ上に展開されたニューラルネットワークの重みやバイアスを直接操作し、ランタイムでニューラルネットワークの誤判断を引き起こさせる。  

### Assessment tools
 * [AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for Attacking Black-box Neural Networks](https://arxiv.org/abs/1805.11770)  
 画像のadversarial example。  
 DNNで訓練された画像認識システムに対するブラックボックスアクセス攻撃を効率化するフレームワーク。  

 * [ADAGIO: Interactive Experimentation with Adversarial Attack and Defense for Audio](https://arxiv.org/abs/1805.11852)  
 音声のadversarial example。  
 音声認識システムの誤認識を誘発するAudio adversarial exampleへの耐性を評価するツール。  

 * [Adversarial Robustness Toolbox for Machine Learning Models](https://www.blackhat.com/us-18/arsenal/schedule/index.html#adversarial-robustness-toolbox-for-machine-learning-models---arsenal-theater-demo-12026)  
 画像のadversarial examplenoの耐性を評価するフレームワーク。  

 * [MLSploit: Resilient ML Platform - Advanced Deep Learning Analytic Platform Made Easy for Every Security Researcher](https://www.blackhat.com/us-18/arsenal/schedule/index.html#mlsploit-resilient-ml-platform---advanced-deep-learning-analytic-platform-made-easy-for-every-security-researcher-11798)  
 画像のadversarial exampleの耐性を評価するフレームワーク。  

 * [Deep Pwning](https://github.com/cchio/deep-pwning)  
 機械学習版のMetasploitを目指したツール。  
 更新は止まっているようだが、コンセプトなどを参考にする。  
