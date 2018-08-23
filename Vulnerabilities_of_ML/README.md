## Vulnerabilities of Machine Learning
安全性評価手法のカテゴリ分類（仮）。  

 * Adversarial example  
 * Decision boundary manipulation  
 * Stealing model and privacy data  
 * Trojan  
 * Unsafety framework and library  
 * Other  

## Vulnerability
### Adversarial example
 MLモデルに入力するデータ（画像や音声等）に微小な細工を加えることで、モデルの判断を誤らせる手法。  
 微小な細工につき、データが細工されているか否か判断することは困難。  

 * [Simple Black-Box Adversarial Perturbations for Deep Networks](https://arxiv.org/abs/1612.06299)  
 画像のAdversarial example。  
 Convolutional Neural Network（以下、CNN）への入力画像を微小に細工することでCNNの判断を誤らせる手法。  
 画素数が小さい画像では1pixel、大きな画像でも数pixelの細工で誤分類させることが可能。  
 [一部検証済み](https://www.mbsd.jp/blog/20170516.html)  

 * [Robust Physical-World Attacks on Deep Learning Models](https://arxiv.org/abs/1707.08945)  
 画像のAdversarial example。  
 自動運転自動車を念頭に、実環境にAdversarial exampleを適用した研究。  
 道路標識に細工を加えることで、画像認識システムの判断を誤らせる手法。  
 [参考記事](https://gigazine.net/news/20170807-robust-physical-perturbations/)  

 * [GenAttack: Practical Black-box Attacks with Gradient-Free Optimization](https://arxiv.org/abs/1805.11090)  
 画像のAdversarial example。  
 遺伝的アルゴリズムを使用して効率良くAdversarial exampleを生成する。  
 既存手法よりも少ないクエリ数で攻撃可能とのこと。  

 * [Targeted Adversarial Examples for Black Box Audio Systems](https://arxiv.org/abs/1805.07820)  
 音声のAdversarial example。  
 音声に微小なノイズを入れて音声認識システムの判断を誤らせる手法。  
 遺伝的アルゴリズムを使用してAdversarial exampleを生成する。  
 検証コードは[こちら](https://github.com/rtaori/Black-Box-Audio)。  

 * [Adversarial Attacks Against Automatic Speech Recognition Systems via Psychoacoustic Hiding](https://arxiv.org/abs/1808.05665)  
 音声のAdversarial example。  
 音声に人間の知覚範囲外のノイズを加えることで、音声認識システムの判断を誤らせる手法。  

 * [Black-Box Attacks against RNN based Malware Detection Algorithms](https://arxiv.org/abs/1705.08131v1)  
 アンチマルウエア版のAdversarial example。  
 RNNベースのアンチマルウエアの判断を誤らせる手法。  

 * [Gradient Band-based Adversarial Training for Generalized Attack Immunity of A3C Path Finding](https://arxiv.org/abs/1807.06752)  
 強化学習版のAdversarial example。  
 最適経路選択を行う強化学習エージェントの判断を誤らせる手法。  
 エージェントに入力される地図データに細工を加えることで、エージェントの経路選択を誤らせる手法。  

 * [Data Poisoning Attacks in Contextual Bandits](https://arxiv.org/abs/1808.05760)  
 強化学習版のAdversarial example。  
 エージェントへの報酬に細工を加えることで、攻撃者が意図した行動をエージェントに行わせる手法。  

### Decision boundary manipulation  
 訓練データを細工することで、MLモデルのDecision boundary（以下、決定境界）を操作する手法。  
 これにより、機械学習ベースのスパムフィルタや侵入検知等を回避することが可能となる。  

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
 モデルが復元されることで、（ユーザにクエリ数で課金してる）クラウドサービスに打撃を与える。  
 また、訓練データに機微情報が含まれている場合、機微情報が漏洩する可能性もある。  

 * [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)  
 MLモデルへの入出力をブラックボックスアクセスで観測し、MLモデルの内部ロジックを復元する手法。  
 2項ロジスティック回帰や多項ロジスティック回帰、多層パーセプトロン、決定木等、多数のMLモデルを標的にしている。  
 検証コードは[こちら](https://github.com/ftramer/Steal-ML)  
 [一部検証済み](https://www.mbsd.jp/blog/20170117.html)  

 * [Machine Learning with Membership Privacy using Adversarial Regularization](https://arxiv.org/abs/1807.05852)  
 TBA.  

### Trojan
 平時はノーマルなMLモデルとして動作し、特定の値を入力した際に攻撃者の意図した動作を行わせる手法。  
 機械学習版のTrojanのようなもの。  

 * [Neural Trojans](https://arxiv.org/abs/1710.00942v1)  
 TBA.  

 * [Hardware Trojan Attacks on Neural Networks](https://arxiv.org/abs/1806.05768)  
 TBA.  

 * [DeepLocker - Concealing Targeted Attacks with AI Locksmithing](https://www.blackhat.com/us-18/briefings/schedule/index.html#deeplocker---concealing-targeted-attacks-with-ai-locksmithing-11549)  
 Deep Neural Network（以下、DNN）を利用した標的型マルウエア。  
 平時はビデオアプリ等として動作し、（顔画像や音声等で）標的を認識するとランサムウエア化する。  
 ランサムウエアのペイロードを暗号化（ロック）しておき、DNNで標的を認識するとその復号鍵を生成して動作するため、既存のアンチマルウエアでは検知が難しいとの事。  

### Unsafety framework and library
 機械学習フレームワークやライブラリに存在する脆弱性を利用する手法。  

 * [Summoning Demons: The Pursuit of Exploitable Bugs in Machine Learning](https://arxiv.org/abs/1701.04739)  
 OpenCVやScikit-Learn等の脆弱性を利用してDoSやBoF等を引き起こす手法。  

 * [Security Risks in Deep Learning Implementations](https://arxiv.org/abs/1711.11008)  
 TensorFlowやCaffe等が利用しているライブラリの脆弱性を利用してDoSやBoF等を引き起こす手法。  

### Other
 上記以外の手法。  

 * [StuxNNet:Practical Live Memory Attacks on Machine Learning Systems](https://aivillage.org/material/cn18-norwitz/slides.pdf)  
 実行中のNeural Network（以下、NN）の判断を誤らせる手法。  
 NN実行中のメモリに展開されたNNの重みやバイアスを直接操作し、NNに誤判断を引き起こさせる。  

 * [Practical Fault Attack on Deep Neural Networks](https://arxiv.org/abs/1806.05859)  
 組み込み機器にビルドインされたDNNの判断を誤らせる手法。  
 機器にレーザを照射し、DNNの隠れ層の計算結果に誤りを生じさせる。  

## Assessment tools
 MLモデルの安全性を評価するためのツール。  

 * [AutoZOOM: Autoencoder-based Zeroth Order Optimization Method for Attacking Black-box Neural Networks](https://arxiv.org/abs/1805.11770)  
 画像のAdversarial exampleに対する耐性を評価。  

 * [Adversarial Robustness Toolbox for Machine Learning Models](https://www.blackhat.com/us-18/arsenal/schedule/index.html#adversarial-robustness-toolbox-for-machine-learning-models---arsenal-theater-demo-12026)  
 画像のAdversarial exampleに対する耐性を評価。  

 * [MLSploit: Resilient ML Platform - Advanced Deep Learning Analytic Platform Made Easy for Every Security Researcher](https://www.blackhat.com/us-18/arsenal/schedule/index.html#mlsploit-resilient-ml-platform---advanced-deep-learning-analytic-platform-made-easy-for-every-security-researcher-11798)  
 画像のAdversarial exampleに対する耐性を評価。  

 * [ADAGIO: Interactive Experimentation with Adversarial Attack and Defense for Audio](https://arxiv.org/abs/1805.11852)  
 音声のAdversarial exampleに対する耐性を評価。  

 * [Deep Pwning](https://github.com/cchio/deep-pwning)  
 機械学習版のMetasploitを目指したツール。  
 更新は止まっているようだが、コンセプト等を参考にしたい。  

## Other contents
 * [ADVERSARIAL MACHINE LEARNING TUTORIAL](https://aaai18adversarial.github.io/)  
 Adversarial攻撃のチュートリアル。防御手法も掲載されている。  
 情報提供：[@icoxfog417](https://twitter.com/icoxfog417) 氏  
 
以上
