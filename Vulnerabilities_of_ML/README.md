## Vulnerabilities of Machine Learning
脆弱性のカテゴリ分類（仮）。  
内容は仮であり、正確性を欠く可能性があります。  

 * Adversarial example  
 * Boundary manipulation  
 * Stealing model and privacy data  
 * Trojan  
 * Unsafety framework and library  
 * Other  

### Adversarial example
 MLモデルに入力するデータ（画像や音声など）に微小な細工を加えることで、モデルの判断を誤らせる手法。  
 微小な細工につき、データが細工されているか否か判断することは困難。  

 * [Simple Black-Box Adversarial Perturbations for Deep Networks](https://arxiv.org/abs/1612.06299)  
 画像のAdversarial example。  
 Convolutional Neural Network（以下、CNN）への入力画像を微小に細工することでCNNの判断を誤らせる手法。  
 画素数が小さい画像では1pixel、大きな画像でも数pixelの細工で誤分類させることが可能。  
 [一部検証済み](https://www.mbsd.jp/blog/20170516.html)  

 * [Robust Physical-World Attacks on Deep Learning Models](https://arxiv.org/abs/1707.08945)  
 画像のAdversarial example。  
 実環境にAdversarial exampleを適用した研究。  
 道路標識に細工を加えることで画像認識システムの判断を誤らせる手法。  

 * [GenAttack: Practical Black-box Attacks with Gradient-Free Optimization](https://arxiv.org/abs/1805.11090)  
 画像のAdversarial example。  
 遺伝的アルゴリズムでAdversarial exampleを生成する。  
 既存手法と比較して、数桁オーダの少ないクエリ数で攻撃可能とのこと。  

 * [Targeted Adversarial Examples for Black Box Audio Systems](https://arxiv.org/abs/1805.07820)  
 音声のAdversarial example。  
 音声に微小なノイズを入れて音声認識システムの判断を誤らせる手法。  
 検証ではMozilla Deep Speechを標的にしている。  
 ソースコードは[こちら](https://github.com/rtaori/Black-Box-Audio)。  

 * [Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding](https://arxiv.org/abs/1808.05665)  
 音声のAdversarial example。  
 オリジナルの音声データに人間の知覚範囲外のノイズを加えることで、音声認識システムの判断を誤らせる手法。  

 * [Black-Box Attacks against RNN based Malware Detection Algorithms](https://arxiv.org/abs/1705.08131v1)  
 アンチマルウエア版のAdversarial example。  
 RNNベースのアンチマルウエアを誤認識させるAdversarial exampleを生成する手法。  

 * [Gradient Band-based Adversarial Training for Generalized Attack Immunity of A3C Path Finding](https://arxiv.org/abs/1807.06752)  
 強化学習版のAdversarial example。  
 最適経路選択を行う強化学習エージェントの判断を誤らせる手法。  
 エージェントにInputされるマップに細工を加えることで、エージェントの経路選択を誤らせる手法。  

 * [Data Poisoning Attacks in Contextual Bandits](https://arxiv.org/abs/1808.05760)  
 強化学習版のAdversarial example。  
 報酬に細工を加えることで、エージェントに攻撃者が意図した行動を行わせる手法。  
 The Multi-armed bandit problemを標的にしている。  

### Decision boundary manipulation  
 訓練データを細工することで、MLモデルのDecision boundary（以下、決定境界）を操作する手法。  
 これにより、機械学習ベースのスパムフィルタや侵入検知などを回避することが可能となる。  

 * [Making & Breaking Machine Learning Anomaly Detectors in Real Life](https://www.slideshare.net/codeblue_jp/making-breaking-machine-learning-anomaly-detectors-in-real-life-by-clarence-chio-code-blue-2015)  
 訓練データに少数の細工データを紛れ込ませることで、決定境界を操作する手法。  
 CODE BLUE 2015でClarence Chio氏により発表された手法。  
 
 * [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)  
 訓練データに少数の細工データを紛れ込ませることで、決定境界を操作する手法。  
 Webやメールボックスから定期的にデータを収集して学習するようなモデルを標的にしている。  

 * [Manipulating Machine Learning: Poisoning Attacks and Countermeasures for Regression Learning](https://arxiv.org/abs/1804.00308)  
 線形回帰モデルを標的にしている。  
 訓練データに細工を加えることで、MLモデルの予測結果を操作する手法。  

### Stealing model and privacy data
 MLモデルの入出力情報から、モデルの内部ロジックや（機微情報を含む可能性のある）訓練データを復元する手法。  
 モデルが復元されることで、ユーザのクエリ数で課金してるクラウドサービスに打撃を与える。  
 また、訓練データに機微情報が含まれている場合、情報漏洩となる可能性もある。  

 * [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)  
 MLモデルへの入出力をブラックボックスアクセスで観測し、MLモデルの内部ロジックを抽出する手法。  
 バイナリロジスティック回帰やマルチロジスティック回帰、多層パーセプトロン、決定木など、多数のMLモデルを標的にしている。  
 [一部検証済み](https://www.mbsd.jp/blog/20170117.html)  

 * [Machine Learning with Membership Privacy using Adversarial Regularization](https://arxiv.org/abs/1807.05852)  
 MLモデル復元の対策。  

### Trojan
 平時はノーマルなMLモデルとして動作し、特定の値を入力した際に攻撃者の意図した動作を行わせる手法。  
 機械学習版のTrojan。  

 * [Neural Trojans](https://arxiv.org/abs/1710.00942v1)  
 ベンダが配布するMLモデルが細工されていることを前提とした攻撃手法。  

 * [Hardware Trojan Attacks on Neural Networks](https://arxiv.org/abs/1806.05768)  
 機械学習版トロイの木馬。  

 * [DeepLocker - Concealing Targeted Attacks with AI Locksmithing](https://www.blackhat.com/us-18/briefings/schedule/index.html#deeplocker---concealing-targeted-attacks-with-ai-locksmithing-11549)  
 DNNを使用した標的型マルウエア。  
 平時はビデオアプリとして動作し、（顔画像や音声などで）標的を認識するとランサムウエア化する。  
 平時はランサムウエアのペイロードを暗号化（ロック）しておき、DNNで標的を認識するとその復号鍵を生成する。  

### Unsafety framework and library
 MLモデルを構築する際に使用されるフレームワークやライブラリに存在する脆弱性を利用する手法。  

 * [Summoning Demons: The Pursuit of Exploitable Bugs in Machine Learning](https://arxiv.org/abs/1701.04739)  
 OpenCVやScikit-Learn等の脆弱性を利用してDoSやBoFなどを引き起こす手法。  

 * [Security Risks in Deep Learning Implementations](https://arxiv.org/abs/1711.11008)  
 TensorFlowやCaffe等が利用しているライブラリの脆弱性を利用してDoSやBoF等を引き起こす手法。  

### Other
 * [StuxNNet:Practical Live Memory Attacks on Machine Learning Systems](https://aivillage.org/material/cn18-norwitz/slides.pdf)  
 実行中のニューラルネットワークに誤判断を引き起こさせる手法。  
 メモリ上に展開されたニューラルネットワークの重みやバイアスを直接操作し、ランタイムでニューラルネットワークの誤判断を引き起こさせる。  

 * [Practical Fault Attack on Deep Neural Networks](https://arxiv.org/abs/1806.05859)  
 組み込み機器にビルドインされたDNNに誤分類を引き起こさせる手法。  
 機器にレーザを照射し、隠れ層の計算結果に誤りを生じさせる。  

## Assessment tools
 MLモデルの安全性を評価するためのツール。  

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

以上
