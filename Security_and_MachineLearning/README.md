# セキュリティエンジニアのための機械学習入門の入門

　今や機械学習は様々なサービスやプロダクトで利用されており、セキュリティ分野でも利用が広がっています。そこで本ブログでは、これまで機械学習に馴染みが無かった**セキュリティエンジニアを対象**に、下記のコンテンツを通して機械学習の利用例と、機械学習の脆弱性及び対策を学ぶことを目的としています。  

 **[English version is here](./README_en.md)**  

### Defensive part
 1. [侵入検知 \(Logistic Regression\)](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap1_IntrusionDetection.md)  
 2. [スパム検知 \(Naive Bayes\)](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap2_SpamDetection.md)  
 3. [異常検知 \(K Nearest Neighbor\)](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap3_AnomalyDetection.md)  
 4. [ログ分析による攻撃検知 \(K means Clustering\)](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap4_AttackDetection.md)  
 5. [脆弱性のトレンド分析 \(Topic model\)](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap5_VulnerabilityTrend.md)  
 6. 脆弱性検査文字列のレコメンド (Multi-Layer perceptron)  
 7. 顔認証 (Convolutional Neural Network)  

　Defensive partは「**機械学習をセキュリティのタスクに利用する例**」を通して機械学習の基本を学びます。  

### Offensive part
 1. 機械学習モデルの窃取 (Model extraction)  
 2. スパム検知モデルのバイパス (Data poisoning)  
 3. 顔認識モデルのバイパス (Adversarial Examples)  
 4. 音声認識モデルを欺く (Audio Adversarial Examples)  
 5. バックドア (Neural Trojan)  
 6. マルウェアの秘匿化 (DeepLocker)  
 7. フェイク動画 (DeepFake)  
 8. フェイク音声 (Audio DeepFake)  

　Offensive partは「**機械学習の脆弱性やサイバー攻撃への利用例および対策**」を学びます。  

　なお、本ブログは、**機械学習入門の入門**の位置付けとなるため、理論（数式）の解説ではなく、実際にコードを動かして**機械学習の利用方法を学ぶ**ことを目的としています。理論からしっかりと学びたい方は、神々が書いたブログや書籍をご参考にして頂ければと思います。  

| Note |
|:-----|
| 本ブログのサンプルコードは、機械学習の利用例を分かり易くすることを念頭に置いた簡易的なコードです。実際のサービスやプロダクトの実装コードとは必ずしも一致しないことを予めご了承ください。|

| Note |
|:-----|
| 「Defensive part」はセキュリティのタスクを機械学習で自動化する例になっています。ここで、自動化の手段は機械学習に限りません。必ずしも「自動化＝機械学習」ではありませんので、自動化のタスクに取り組む際は、目的に応じて適切な技術を採用されることを推奨いたします。|

### CHANGELOG
2018.09.25(Tue): [1.侵入検知](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap1_IntrusionDetection.md)の初稿を公開。  
2018.11.15(Thu): [2.スパム検知](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap2_SpamDetection.md)の初稿を公開。  
2018.12.30(Sun): [3.異常検知](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap3_AnomalyDetection.md)の初稿を公開。  
2019.01.09(Wed): [4.ログ分析による攻撃検知](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap4_AttackDetection.md)の初稿を公開。  
2019.06.20(Thu): コンテンツの一部を変更（「全自動侵入エージェント」から「Neural Trojan」に変更）。  
2019.06.24(Mon): コンテンツに「Webアプリへの攻撃検知」と「攻撃検知のBypass/対策」を追加。  
2019.10.03(Thu): [5.脆弱性のトレンド分析](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Security_and_MachineLearning/Chap5_VulnerabilityTrend.md)の初稿を公開。  
