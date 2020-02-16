# Transferable Clean-Label Poisoning Attacks on Deep Neural Nets
[original paper](https://arxiv.org/abs/1905.05897)  

## Abstract
Clean-label poisoning攻撃は、一見すると無害に見える**汚染画像**をデータセットに注入し、汚染データセットで学習したモデルに**特定の入力画像を攻撃者の意図したクラスに分類**させる攻撃である。本論文では、攻撃対象モデルの出力・アーキテクチャ・データセットが未知の状態（ブラックボックス）で攻撃を行う**Polytope攻撃**を提案する。Polytope攻撃では、特徴空間において標的とする画像（以下、標的画像）を汚染画像で囲むように配置する。また、汚染画像の作成時に**Dropout**を適用することで、攻撃の汎用性が高まることも示す。本手法では、僅か1％のデータセットを汚染するだけで、攻撃の成功率を50％以上にすることが可能である。  

## 1. Introduction
Deep Neural Network（以下、DNN）では、モデルを学習させるための大規模なデータセットが必要となる。このため、多くの開発者は学習データをWebからスクレイピングなどで自動収集することが多い。しかし、最近の研究では、このようなデータ収集方法が脆弱性に繋がることが実証されている。特に、信頼できないソースからデータを収集した場合、作成したモデルがpoisoning攻撃に対して脆弱になり、攻撃者にモデルが乗っ取られる可能性が高まる（攻撃者の意図した動作を行う）。  

本論文では、画像分類器に対する効率的で汎用的なClean-label poisoning攻撃を提案する。一般的なpoisoning攻撃は、データセットに変更を加えることで、**モデルの推論動作を制御**することを目的とする。従来の回避攻撃やバックドア攻撃とは対照的に、我々は攻撃対象モデルの推論中にデータセットを変更しないケースを検証する。  

Clean-label poisoning攻撃は、従来のpoisoning攻撃とは大きく異なる。それは、データセットの**ラベル付け作業に攻撃者が関与する必要がない**ことである。よって、汚染画像は開発者によって正しくラベル付けされた場合でも、誤分類を引き起こす**摂動**を保持することが求められる。このような攻撃手法は、攻撃者がWeb上に汚染画像を**ばら撒く**だけでデータセットを汚染することができるため、Webスクレイピングや（汚染に気づかない）開発者によって汚染画像が収集されるのを待つだけで良い。そして、収集された汚染画像は開発者によって正しくラベル付けされ、データセットに挿入される。更に、本手法はモデルの分類精度をむやみに低下させるのではなく、特定クラスのみをターゲットにすることが可能である。つまり、モデル全体の精度が低下しないため、本攻撃の検知は困難となる。  

従来のClean-label poisoning攻撃は、攻撃者が攻撃対象モデルに対する知識を持ち、汚染画像を作成する過程でこの知識を活用するという**ホワイトボックス**な状態でのみ実証されている。一方、我々は攻撃者が攻撃対象モデルに対する知識を持たない**ブラックボックス**な状態で汚染画像を作成する手法を提案する。  

ブラックボックスのClean-label poisoning攻撃は、以下2つの理由で非常に挑戦的なタスクである。  
 
 * 汚染データセットを基に作成される攻撃対象モデルの決定境界（Decision Boundary）は未知である。  
 * 攻撃者は攻撃対象モデルにアクセスできないため、攻撃対象モデルに依存せずに汚染画像を作成する必要がある。  

本論文では、汎用性の高いClean-label poisoning攻撃を提案する。本攻撃の前提条件として、攻撃者は攻撃対象モデルの出力やハイパーパラメータにアクセスできないが、**攻撃対象モデルと同等のデータセットを収集できる**こととする。攻撃者はこのデータセットで**代替モデルを作成**し、特徴空間において**標的画像の特徴をConvex内部に閉じ込める**ように汚染画像を作成する。そして、汚染画像を過学習した攻撃対象モデルは、**標的画像を汚染画像と同じクラスに分類**する。本手法の特徴を纏めると以下の通りとなる。  

 * ブラックボックスで攻撃可能である。  
 * 中間層に本攻撃を適用することで、従来のブラックボックス攻撃（Feature Collision攻撃）よりも成功率が高くなる。  
 * 転移学習モデルとEnd-to-Endモデルの両方で攻撃成功率が高い。  
 * 汚染画像の作成時にDropoutを使用すると、攻撃の汎用性が向上する。  

## 2. The Threat Model
我々は、僅かな汚染画像を攻撃対象モデルが使用するデータセットに注入できる攻撃者を想定する。そして、攻撃者の目的は、汚染データセットを学習した攻撃対象モデルに、攻撃者が意図したクラスにテストデータを分類させることである。  

攻撃対象モデルへの完全なアクセス、またはクエリアクセスを必要とするShafahiやPapernotらの研究とは異なり、我々は自動運転自動車や監視システムのように、攻撃者が攻撃対象モデルにアクセスできないケースを想定する。その代わりに、攻撃者は攻撃対象モデルのデータセットに関する知識を有し、（汚染画像を作成するために使用される）代替モデルのデータセットとして攻撃対象モデルと同様のデータセットを収集できることを前提とする。  

我々は、攻撃対象モデルが採用する学習方法として以下2つを検討する。  

 * 転移学習  
 転移学習では、CIFARやImageNetなどの公開データセットで学習した特徴抽出器`φ`をベースとし、これに新たなデータセット`X`を学習させることでファインチューニングする。転移学習は、モデル作成時に大規模なデータセットが利用できない産業用アプリケーションなどでは一般的に利用されている。転移学習モデルに対する従来のpoisoning攻撃は、KohとLiangらやShafahiらの研究のように、特徴抽出器の内部構造が明らかになっているホワイトボックス設定でのみ検証されている。  

 * End-to-End学習  
 標的画像に加えた摂動は、特徴抽出器のハイパーパラメータ（重みやバイアスなど）の最適化に影響を与えるため、転移学習よりも摂動に対して厳しい制約がある。Shafahiらは、10クラスの画像分類問題で約60％の攻撃成功率を達成するために、標的画像に最大30％、約40個の汚染画像に重ね合わせるという**透かし戦略**を採用した。  

## 3. Transferable Targeted Poisoning Attacks
### 3.1. Difficulties of Targeted Poisoning Attacks
標的型poisoning攻撃において、攻撃者は標的画像`xt`を（攻撃者が意図した）クラス`y~t`に分類させるような攻撃対象モデルを取得する必要がある。1つの簡単なアプローチは、クラス`y~t`の画像から汚染画像を作成し、特徴空間において汚染画像を標的画像`xt`に近づけることである。学習時に損失が飽和した後でも汎化は改善し続けることが実証されているため、攻撃対象モデルはデータセットに過学習する。結果として、特徴空間において汚染画像が標的画像に近い場合、汚染画像の近くの空間は攻撃者が意図したクラス`y~t`として分類されるため、攻撃対象モデルが`xt`を`y~t`に分類する可能性が高まる。しかし、図1に示すように、汚染画像と標的画像の距離が近くても攻撃が成功する保証はない。実際、汚染画像を標的画像に近づけることは、非常に多くの制約が存在する。  

 <div align="center">
 <figure>
 <img src='./img/Fig1.png' alt='Figure1'><br>
 <figurecaption>Figure 1. An illustrative toy example of a linear SVM trained on a two-dimensional space with training sets poisoned by Feature Collision Attack and Convex Polytope Attack respectively. The two striped red dots are the poisons injected to the training set, while the striped blue dot is the target, which is not in the training set. All other points are in the training set. Even when the poisons are the closest points to the target, the optimal linear SVM will classify the target correctly in the left figure. The Convex Polytope attack will enforce a small distance of the line segment formed by the two poisons to the target. When the line segment’s distance to the target is minimized, the target’s negative margin in the retrained model is also minimized if it overfits.
 </figurecaption><br>
 <br>
 </figure>
 </div>

### 3.2. Feature Collision Attack
Shafahiらによって提案されたFeature Collision攻撃は、ホワイトボックスな状態で汚染画像を生成する手法である。攻撃者は汚染画像`xp`を作成するため、標的とするクラスからベース画像`xb`を選び、`xb`に微小な摂動を加えることで`xp`を特徴空間における標的画像`xt`に近づける。具体的には、攻撃者は以下の最適化問題(1)を解いて汚染画像を作成する。  

 <div align="center">
 <figure>
 <img src='./img/Formula1.png' alt='Formula1'><br>
 <br>
 </figure>
 </div>

ここで、`φ`は事前学習された特徴抽出器である。  
数式の第一項は、入力空間のベース画像の近くに汚染画像を近付けるために、ラベル付けを行う開発者がベース画像`xb`と同じラベルを汚染画像に付けさせるようにしている。そして、第二項は、汚染画像と標的画像の特徴量を強制的に衝突させている。ハイパーパラメーター`µ > 0`は、これらの項のバランスをトレードオフで調整するために使用される。  

汚染画像`xp`が標的クラスとしてラベル付けされ、攻撃対象モデルのデータセット`X`に注入された場合、攻撃対象モデルは汚染画像の特徴量を標的クラスに分類するように学習する。次に、`xt`と`xp`の特徴量が近い場合、`xt`は`xp`と同じクラスに分類される（攻撃成功）。より攻撃成功率を高めるために、複数の汚染画像が使用される場合もある。  

残念ながら、異なる特徴抽出器`φ`は異なる特徴空間を持つため、とある特徴空間の距離`d_φ(x_p, x_t) = ||φ(x_p) − φ(x_t)||`が別の特徴空間の距離と一致するとは限らない。しかし、幸運なことにTramerらの研究結果は、各入力画像について、**同じデータセットで学習された異なるモデルには敵対部分空間が存在する**ことを示している。このため、ベース画像に適度な摂動を加えることにより、各モデルがサンプルを誤分類する可能性がある。これは、モデルが同じデータセット、または**類似のデータセットで学習されている場合、異なるモデルの特徴空間で汚染画像`xp`を標的画像`xt`に近付けることが可能**であることを示している。  

よって、ブラックボックス攻撃を行うための最も明白なアプローチは、汚染画像のセット`{x_p^(j)}_j = 1^k`を最適化し、複数のモデル`{φ^(i)}_i = 1^m`で汚染画像と標的画像の特徴量を近づけるような汚染画像を生成することである（`m`は使用するモデルの数）。このような手法は、ブラックボックスの回避攻撃でも使用されている。異なる特徴抽出器は、異なる次元・大きさの特徴ベクトルを生成するため、次のように正規化する。  

 <div align="center">
 <figure>
 <img src='./img/Formula2.png' alt='Formula2'><br>
 <br>
 </figure>
 </div>

### 3.3. Convex Polytope Attack
Feature Collision攻撃の問題点の1つは、ベース画像に加えた摂動が人間に認識され易いことである。クロスエントロピーのような単一エントリの損失を最大化する回避攻撃とは異なり、Feature Collision攻撃は汚染画像の特徴ベクトル`φ(xp)`の各エントリを`φ(xt)`に近づけるため、各汚染画像`xp`には数百～数千の制約が生じる。さらに、数式(2)におけるブラックボックス設定では、`m`個のモデルにおいて衝突を行うため、汚染画像に対する制約数は更に増えることになる。このような多数の制約が存在するため、Optimizerはベース画像を標的画像に近付けるため、`xp`を標的クラスのように見せる。その結果、ラベル付けを行う開発者は画像の違和感に気づき、汚染画像をデータセットから除外する可能性が高くなる。図9は、フックの画像から汚染画像を作成し、数式(2)で魚の画像を攻撃する処理過程を示している。  

 <div align="center">
 <figure>
 <img src='./img/Fig2.png' alt='Figure2'><br>
 <figurecaption>Figure 2. A qualitative example of the difference in poison images generated by Feature Collision (FC) Attack and Convex Polytope (CP) Attack. Both attacks aim to make the model mis-classify the target fish image on the left into a hook. We show two of the five hook images that were used for the attack, along with their perturbations and the poison images here. Both attacks were successful, but unlike FC, which demonstrated strong regularity in the perturbations and obvious fish patterns in the poison images, CP tends to have no obvious pattern in its poisons. More details are provided in the supplementary.
 </figurecaption><br>
 <br>
 </figure>
 </div>

Feature Collision攻撃のもう1つの問題点は汎用性の欠如である。Feature Collision攻撃はブラックボックスな状態では失敗する傾向がある。これは、特徴空間内の全てのモデルで汚染画像を`xt`に近づけることが困難であるためである。ニューラルネットワークは1つの線形分類器を使用して特徴`φ(x)`を分離するため、異なる特徴抽出器の特徴空間は大きく異なる。よって、汚染画像`xp`が代替モデルの特徴空間内で`xt`と衝突する場合でも、未知の攻撃対象モデルでは`xp`と`xt`が衝突しない可能性がある。図1に示すように、`xp`と`xt`の距離が短い場合でも攻撃は失敗する可能性がある。また、多くの代替モデルを使用してこのようなエラーを減らすことは実用的ではない。  

よって、我々は汚染画像に不自然な模様が表れず、汎化の制約が軽減されるように、汚染画像に対する緩い制約を求める必要がある。Shafahiらの研究は、複数の汚染画像を使用して1つの標的画像を攻撃する。先ず、汚染画像の特徴ベクトルのセット`{φ(x_p^(j))}_j = 1^k`の必要十分条件を導き出す。そのため、標的画像`xt`は汚染画像のクラスに分類される。  

 * Proposition 1. The following statements are equivalent:  
 <div align="center">
 <figure>
 <img src='./img/Proposition1.png' alt='Proposition1'><br>
 <br>
 </figure>
 </div>

Proposition1の証明は、巻末の補足資料に記載している。  
同じクラスの汚染画像セットは、標的画像の特徴ベクトルが汚染画像の特徴ベクトルの凸多面体内にある場合、標的画像のクラスを汚染画像のクラスに変更することが保証される。これは、特徴の衝突を強制するFeature Collision攻撃よりもかなり緩和された制約である。これにより、大きな領域のクラスを変更しながら、汚染画像を標的画像から遠くに配置することが可能となる。`xt`が攻撃対象モデル内の凸多面体内に存在し、`{x_p^(j)}`が期待通りに分類される場合に限り、`xt`は`{x_p^(j)}`と同じクラスとして分類される。  

この想定により、標的画像の特徴ベクトルが凸多面体内、またはその近くに配置されるように、特徴空間に凸多面体を形成するような汚染画像セットを最適化する。具体的には以下の最適化問題を解く。  

 <div align="center">
 <figure>
 <img src='./img/Formula3.png' alt='Formula3'><br>
 <br>
 </figure>
 </div>

ここで、`x_b^(j)`は`j-th`の汚染画像に対するベース画像であり、`ε`は（人間が知覚できない程度の）摂動の最大許容量である。  

数式(3)は、標的画像が`m`モデルの特徴空間内の（汚染画像の）凸多面体の中、またはその近くに配置されるように、汚染画像セット`{x_p^(j)}`と凸多面体の組み合わせ係数`{c_j^(i)}`のセットを同時に見つける。係数`{c_j ^（i）}`が解かれることに注意されたい。つまり、モデル毎に異なることが許容されるため、頂点`{x_p^(j)}`を含む凸多面体内の特定の点に近い必要はない。同じ量の摂動が与えられた場合、数式(2)は数式(3)の特殊なケースとなるため、`c_j^(i) = 1/k`を修正するため、そのような目標はFeature Collision攻撃（数式(3)）よりも緩和される。その結果、図9に示すように、汚染画像は殆ど摂動の模様を示さず、Feature Collision攻撃と比較して攻撃の秘匿性が向上する。  

凸多面体の最も大きな利点は汎用性の向上である。  
Convex Polytope攻撃の場合、`xt`は未知の攻撃対象モデルの特徴空間の特定の点に合わせる必要はなく、汚染画像で形成した凸多面体内に存在しさえすれば良い。仮にこの要件が満たされていない場合でも、Convex Polytope攻撃にはFeature Collision攻撃よりも利点がある。与えられた攻撃対象モデルについて、`ρ`より小さな残差1が攻撃の成功を保証すると仮定する。Feature Collision攻撃の場合、標的画像の特徴は、`φ^(t) (x_p^(j*))`を中心とする`ρ-ball`内に存在する必要がある（`j* = argmin j||φ^(t) (x_p^(j)) − φ^(t) (xt)||`）。Convex Polytope攻撃の場合、標的画像の特徴は、`{φ^(t) (x_p^(j))}_j = 1^k`によって形成される凸多面体の`ρ-expansion`にある可能性がある。これは、前述の`ρ-ball`よりも大きいボリュームを持ち、より大きな汎化エラーを許容する。  

### 3.4. An Efficient Algorithm for Convex Polytope Attack
`{φ^(i)}`の複雑さと`{c^(i)}`の凸多面体制約によって生じる困難を回避する代替手法として、非凸型の制約問題(3)を最適化する。`{x_p^(j)}_j = 1^k`が与えられた場合、GoldsteinらによるForwardbackward splittingと呼ばれる手法を使用し、係数`{c^(i)}`の最適なセットを見つける。`c^(i)`の次元は通常は小さいため（一般的な値は5）、この処理はニューラルネットワークの逆伝播よりも遥かに少ない計算で済む。次に、`{x_p^(j)}`に関して最適な`{c^(i)}`が与えられると、`m`個のネットワークを介した逆伝搬は比較的高コストになるため、1つの勾配ステップを使用して`{x_p^(j)}`を最適化する。 最後に、汚染画像をベース画像の`ε`単位内に与えるため、摂動は目立たず、クリップ操作として実装される。アルゴリズム1に示すように、この処理を繰り返すことで、汚染画像と係数の最適なセットを見つける。  

 <div align="center">
 <figure>
 <img src='./img/Formula.png' alt='Formula'><br>
 <br>
 </figure>
 </div>

我々の検証実験では、最初のイテレーションの後、`{c^(i)}`を最後のイテレーションからの値に初期化すると、収束が加速することが分かった。また、攻撃対象ネットワークにおける損失は、momentum optimizerなしで大きな変動をもたらすことが分かる。よって、摂動のoptimizerとしてAdamを使用することにする。Adamを使用した場合、摂動はSGDよりも確実に収束する。摂動の制約は`ℓ∞`ノルムであるが、DongやFGSMなどの摂動を作成するための一般的な手法とは対照的である。これは、更新ステップが既に小さい場合、符号の反転によって引き起こされる分散を低減する。  

 <div align="center">
 <figure>
 <img src='./img/Algorithm1.png' alt='Algorithm1'><br>
 <br>
 </figure>
 </div>

### 3.5. Multi-Layer Convex Polytope Attack
攻撃対象モデルがが分類子（最後の出力層）に加え、特徴抽出器`φ`を学習する場合、検証実験で示すように、`φ^(i)`の特徴空間のみにConvex Polytope攻撃を適用するだけでは不十分である。この設定では、汚染画像で学習されたモデルによって引き起こされる特徴空間の変化も、多面体の汎用性をより困難にする。  

線形分類器とは異なり、DNNの方が画像データセットの汎化が優れている。汚染画像は全て1つのクラスに由来するため、ネットワーク全体が`xt`と`xp`の分布を十分に区別することができる。よって、汚染画像を含むデータセットで学習した後でも`xt`は正しく分類される。図7に示すように、Convex Polytope攻撃（以下、CP攻撃）が出力層に適用されると、`SENet18`や`ResNet18`などの低容量モデルは、他の大容量モデルよりも汚染画像の影響を受け易くなる。しかし、Tramerらの研究によると、同じデータセットで学習された異なるモデルには共通の敵対的部分空間が存在することが分かっているため、汚染画像の分布で学習されたネットワークに攻撃可能な汚染画像が存在する可能性がある。また、自然に訓練されたネットワークは、通常誤分類を引き起こすのに十分な大きさのLipschitzを持ち、これは多面体をそのような部分空間にシフトして`xt`の近くに配置できることが期待される。

汚染画像を含むデータセットを使用してEnd-to-Endで学習されたモデルへの汎用性を高めるための1つの戦略は、ネットワークの複数の層にCP攻撃を適用することである。層の深いネットワークにおいては、層を浅いネットワーク`φ1,...,φ_n`に分割され、Objectiveは次のようになる。  

 <div align="center">
 <figure>
 <img src='./img/Formula4.png' alt='Formula4'><br>
 <br>
 </figure>
 </div>

ここで、`φ_1:l^(i)`は、`φ_l^(i)`から`φ_l^(i)`への連結である。`ResNet`のようなネットワークは、Pool層によって分離されたブロックに分割され、`φ_l^(i)`をブロックのl番目の層とする。`φ1:l(l < n)`までの特徴量で学習された最適な線形分類器は、`φ`の特徴量で学習された最適な線形分類器よりも汎化性能が悪いため、`xt`の特徴量は学習後に同じクラスの特徴量から逸脱する可能性が高くなる。これは、攻撃を成功させるために必要な条件である。一方、このような目的では、摂動は深さの異なるモデルを騙すために最適化される。これにより、代替モデルの多様性がさらに増し、汎用性が向上する。  

### 3.6. Improved Transferability via Network Randomization
同じデータセットで学習した場合でもモデル毎に精度が異なるため、特徴空間でのサンプルの分布も異なる。理想的には、攻撃対象モデルの関数クラスから任意の多くのネットワークを使用して汚染画像を作成する場合、攻撃対象モデルにて数式(3)を効果的に最小化できるはずである。しかし、メモリの制約により、多数のネットワークを組み合わせることは実用的ではない。  

多数のモデルを組み合わせる代わりに、汚染画像の生成時にDropoutを使用してネットワークをランダム化する。各イテレーションにて、各置換ネットワーク `φ^(i)`は、確率`p`で各ニューロンを遮断し、全ての「オン」ニューロンに `1/(1 − p)`を掛けて期待値を変更しないことにより、関数クラスからランダムにサンプリングされる。 このようにして、メモリを節約しながら指数関数的（な深さ）の異なるネットワークを取得することができる。このようにランダム化されたネットワークは、実証実験での汎用性を向上させる。1つの定性的な例を図3に示す。  

 <div align="center">
 <figure>
 <img src='./img/Fig3.png' alt='Figure3'><br>
 <figurecaption>Figure 3. Loss curves of Feature Collision and Convex Polytope Attack on the substitute models and the victim models, tested using the target with index 2. Dropout improved the minimum achievable test loss for the FC attack, and improved the test loss of the CP attack significantly.
 </figurecaption><br>
 <br>
 </figure>
 </div>

 <div align="center">
 <figure>
 <img src='./img/Fig4.png' alt='Figure4'><br>
 <figurecaption>Figure 4. Qualitative results of the poisons crafted by FC and CP. Each group shows the target along with five poisons crafted to attack it, where the first row is the poisons crafted with FC, and the second row is the poisons crafted with CP. In the first group, the CP poisons fooled a DenseNet121 but FC poisons failed, in the second group both succeeded. The second target’s image is more noisy and is probably an outlier of the frog class, so it is easier to attack. The poisons crafted by CP contain fewer patterns of xt than with FC, and are harder to detect.  
 </figurecaption><br>
 <br>
 </figure>
 </div>

## 4. Experiments
以下では、CPとFCをそれぞれConvex Polytope攻撃とFeature Collision攻撃の略語として使用する。実験のコードは、[我々のgithub](https://github.com/zhuchen03/ConvexPolytopePosioning)で入手することができる。

### Datasets
この節では、全ての画像は`CIFAR10`データセットから取得される。明示的に指定されていない場合、攻撃対象モデルと代替モデル`(φ^(i))`を事前学習するために、学習用データセットの10クラスのそれぞれから最初の4,800画像（合計48,000枚の画像）を取得する。テストデータセットはそのままにし、異なる設定でのこれらのモデルの精度を標準テストセットで評価し、ベンチマークと直接比較できるようにする。攻撃が成功する場合、細工の目立たない画像の摂動だけでなく、ベース画像を基に作成された汚染画像を含むデータセットをファインチューニングした後のテスト精度も変わらないはずである。補足で示すように、攻撃は対応するベース画像を含むデータセットで調整されたモデルの精度と比較し、攻撃対象モデルの精度を維持する。

学習用データセットの残りの2,000枚の画像は、攻撃対象画像の選択、汚染画像の作成、および攻撃対象モデルのファインチューニングのためのバックアップとして使用する。このバックアップ内の各クラスの最初の50個の画像（合計500枚の画像）をクリーンなファインチューニング用のデータセットとして取得する。これは、`Imagenet`のような大規模なデータセットの事前学習済みモデルが、類似しているが通常はバラバラのデータセットでファインチューニングされるシナリオに似ている。攻撃対象のクラスとして「船」を、攻撃対象画像のクラスとして「カエル」をランダムに選択する。つまり、攻撃者は**特定のカエル画像を船として誤分類**させたいと考えるシナリオとする。実験で使用する全ての汚染画像`x_p^(j)`は、500枚の画像のファインチューニング用のデータセットの「船」クラスの最初の5画像から作成される。「カエル」クラスの次の50枚の画像で汚染の有効性を評価する。これらの各画像は、統計を収集するターゲット`xt`として個別に評価される。繰り返すが、攻撃対象画像は学習データセットおよびファインチューニング用のセットに含まれていない。

### Networks
この論文では、2組の代替モデルを使用する。セット`S1`には、`SENet18`、`ResNet50`、`ResNeXt29-2x64d`、`DPN92`、`MobileNetV2`、および`GoogLeNet`が含まれている。セット`S2`には、`MobileNetV2`と`GoogLeNet`を除く`S1`の全てのモデルが含まれている。`S1`と`S2`は、以下で指定する様々な実験で使用される。ブラックボックスモデルのアーキテクチャとして、`ResNet18`と`DenseNet121`を使用する。汚染画像は、代替モデルで作成される。6つの異なる代替モデルと2つのブラックボックスモデルの攻撃対象モデルに対するpoisoning攻撃を評価する。ただし、各攻撃対象モデルは、代替モデルとは異なるランダムシードで学習される。代替モデルに攻撃対象モデルが表示される場合、グレーボックス設定とする。それ以外は全てブラックボックス設定である。

`GoogLeNet`の各`Inception`ブロックの出力、および他のネットワークでは各`Residual-like`ブロックの畳み込み層の中央にDropout層を追加する。これらのモデルは、パブリックな[リポジトリ](https://github.com/kuangliu/pytorch-cifar)の同じアーキテクチャとハイパーパラメータ（Dropout除く）を使用し、`0、0.2`、`0.25`、`0.3`のDropout率で前述の48,000画像のデータセットで1から学習する。なお、本検証における攻撃対象モデルはDropoutでは学習されない。

### Attacks
同じ5つの汚染「船」画像を使用し、各「カエル」画像を攻撃する。  
代替モデルには、各モデルにて`0.2`、`0.25`、`0.3`のDropout率で学習された3つのモデルを使用する。これにより、`S1`および`S2`のそれぞれ`18`および`12`の代替モデルが得られる。汚染画像を作成する際、代替モデルの訓練時と同じDropout率を使用する。全ての検証実験で、`ε=0.1`を設定する。モデルは学習データセットに似た画像上に小さな勾配を持つように学習されているため、汚染画像の作成には`0.04`の比較的大きな学習率とし、Optimizerには`Adam`を使用する。各実証実験で汚染の摂動に対して4,000回以下のイテレーションを実行する。なお、特に指定がない限り、最後の層のに対してのみ数式(3)を適用する。  

攻撃対象モデルについては、ファインチューニング中にハイパーパラメータを選択し、前述の攻撃対象モデルの仮定を満たす**500画像の学習データセットを過学習させる**。最後の層の線形分類器のみがファインチューニングされる転移学習では、過学習するために`0.1`という大きな学習率とし、Optimizerには`Adam`を使用する。End-to-Endの設定では、`10^-4`の小さな学習率とし、Optimizerには`Adam`を使用して過学習させる。

### 4.1. Comparison with Feature Collision
先ず、FCとCPで生成された汚染画像の転移学習の設定にて比較する。実験結果を図5に示す。この実験では、代替モデルのセット`S1`を使用する。FCが`0.5`を超える成功率を達成することはなかったが、その一方でCPは`0.5`以上の成功率を達成した。2つのアプローチで作成された汚染画像の一例を図4に示す。  

 <div align="center">
 <figure>
 <img src='./img/Fig5.png' alt='Figure5'><br>
 <figurecaption>Figure 5. Success rates of FC and CP attacks on various models. Notice the first six entries are the gray-box setting where the models with same architecture but different weights are in the substitute networks, while the last two entries are the black-box setting.
 </figurecaption><br>
 <br>
 </figure>
 </div>

### 4.2. Importance of Training Set
FCよりも攻撃成功率が高いにも関わらず、攻撃対象モデルの学習データセットに関する知識がない場合、CPの信頼性について疑問が残る。  
最後の節では、攻撃対象モデルと同じ学習データセットを使用して代替モデルを学習させる。図6は、代替モデルのデータセットが攻撃対象モデルのデータセットに類似（但し、殆どは異なる）している場合の結果を示す。このような実験設定は、攻撃対象モデルの知識が不要な設定よりも現実的であるが、攻撃対象モデルへのクエリアクセスが必要である。クエリアクセスは監視などのシナリオでは利用できないためである。4つの異なるアーキテクチャから作成された12の代替モデルを保持する`S2`を使用する。結果は、転移学習の設定で評価される。学習データが重複していなくても、CPはブラックボックス設定の代替モデルとはかなり異なる構造を持つモデルに汎化できる。学習データの重複率が0％の設定では、`SEPN18`や`MobileNetV2`などの低容量モデルよりも、`DPN92`や`DenseNet121`などの大容量のモデルに汚染がより効率よく転送される。全体として、攻撃対象モデルが汎化されている限り、転移学習設定の学習データセットにアクセスしなくとも、CPは効率よくpoisonin攻撃を行えることが分かる。

 <div align="center">
 <figure>
 <img src='./img/Fig6.png' alt='Figure6'><br>
 <figurecaption>Figure 6. Success rates of Convex Polytope Attack, with poisons crafted by substitute models trained on the first 2400 images of each class of CIFAR10. The models corresponding to the three settings are trained with samples indexed from 1201 to 3600, 1 to 4800 and 2401 to 4800, corresponding to the settings of 50%, 50% and 0% training set overlaps respectively.
 </figurecaption><br>
 <br>
 </figure>
 </div>

### 4.3. End-to-End Training
より困難な設定は、攻撃対象モデルがEnd-to-Endの学習を行う場合である。より汎化されたモデルがより脆弱であることが判明した転移学習の設定とは異なり、ここで適切な汎化は、汚染画像にも関わらずこのようなモデルがターゲットを正しく分類するのに役立つ。図7に示すように、最後の層に対するCPは、汎用性にとって十分ではないため、攻撃は殆ど成功しない。`ResNet50`の成功率が0％であり、これが代替モデルのアーキテクチャであるのに対し、`ResNet18`の成功率は最高であり、代替モデルのアーキテクチャではないが、汎化が悪いはずである。

よって、代替モデルの複数の層でCPを実施する。これにより、モデルがより小さい容量のモデルに分割されるため、より良い結果が得られる。グレーボックス設定では、全ての攻撃が`0.6`を超える成功率を達成した。ただし、`GoogLeNet`に攻撃を行うことは困難なままである。`GoogLeNet`は、代替モデルよりもアーキテクチャが異なる。よって、凸型多面体をEnd-to-Endの学習に適用する方向を見つけることは更に困難である。

 <div align="center">
 <figure>
 <img src='./img/Fig7.png' alt='Figure7'><br>
 <figurecaption>Figure 7. Success rates of Convex Polytope Attack in the end-toend training setting. We use S2 for the substitute models.
 </figurecaption><br>
 <br>
 </figure>
 </div>

## 5. Conclusion
纏めると、我々は標的型のpoisoning攻撃の汎用性を高めるアプローチを提案した。主な貢献は、特徴空間の標的画像の周りに凸多面体を構築する新しい方法の開発である。そのため、汚染されたデータセットに過学習する線形分類器は、攻撃対象画像を汚染画像のクラスに分類することができる。汎用性を改善する2つの実用的な方法を提案した。先ず、汚染画像の作成時にDropoutを使用することである。これにより、異なる構造を持つ様々なネットワーク（アンサンブル）から多様な汚染画像が得られり。第二に、複数の層で凸型多面体を適用する。これにより、End-to-End学習の設定でも攻撃が成功することが判明した。更に、汎用性は、代替モデルの学習データセットに使用されるデータ分布に依存することが判明した。

## A. Proof of Proposition 1
`2 ⇒ 1`  

多クラス分類問題の場合、`φ(x)`が`ℓ_p`として分類される条件は次のとおりである。  

 <div align="center">
 <figure>
 <img src='./img/A-1.png' alt='FormulaA-1'><br>
 <br>
 </figure>
 </div>

これらの各制約は線形であり、凸半空間によって満たされる。凸半空間の交点でこれらの制約の全てを満たす空間、すなわち凸領域。 条件(2)の下では、`φ(xt)`はこの凸領域内の点の凸の組み合わせであるため、`φ(xt)`自体はこの凸領域内にある。

`2 ⇒ 1`  

数式(1)が成り立つと仮定する。

 <div align="center">
 <figure>
 <img src='./img/A-2.png' alt='FormulaA-2'><br>
 <br>
 </figure>
 </div>

数式(2)を点`φ(x_p^j)_j = 1^k`の凸包とする。`t = argmin_u ∈ S||u−φ(x_t)||`を `S`の`φ(x_t)`に最も近い点とする。`||u_t−φ(x_t)|| = 0`の場合、(2)が成立し、証明が完了する。`||u_t−φ(x_t)|| > 0`の場合、分類関数を定義する。

 <div align="center">
 <figure>
 <img src='./img/A-3.png' alt='FormulaA-3'><br>
 <br>
 </figure>
 </div>

明らかに `f(φ(x_t))<0`。条件(1)により、`f(φ(x_p^j)) < 0`の`j`もある。関数を考える。 

 <div align="center">
 <figure>
 <img src='./img/A-4.png' alt='FormulaA-4'><br>
 <br>
 </figure>
 </div>

`ut`は`S`の`φ(x_t)`に最も近く、`g`は滑らかなため、`η`に対する`g`の導関数は`η = 0`で評価され`0`である。この微分条件を次のように書くことができる。

 <div align="center">
 <figure>
 <img src='./img/A-5.png' alt='FormulaA-5'><br>
 <br>
 </figure>
 </div>

しかし、`f(φ(x_p^j)) < 0`であるため、この記述は矛盾している。

## B. Comparison of Validation Accuracies
データのpoisoning攻撃を検出し難くするために、非自明な摂動を行うことに加え、汚染画像を含むデータセットでファインチューニングされたモデルの精度は、同じ（汚染画像を除く）クリーンなデータセットでのファインチューニングと比較して大幅に低下しない。図8は、精度の低下が実際に現れないことを示している。

## C. Details of the qualitative example
標的となる「魚」の画像と、汚染画像の作成に使用される5つの「フック画像」は、`ImageNet`と同じクラスを持つ`WebVision`データセットから取得する。図9は、5つの汚染画像を示している。

 <div align="center">
 <figure>
 <img src='./img/Fig8.png' alt='Figure8'><br>
 <figurecaption>Figure 8. Accuracies on the whole CIFAR10 test for models trained or fine-tuned on different datasets. The fine-tuned models are initialized with the network trained on the first 4800 images of each class. There is little accuracy drop after fine-tuned on the poisoned datasets, compared with fine-tuning on the clean 500-image set directly.
 </figurecaption><br>
 <br>
 </figure>
 </div>

 <div align="center">
 <figure>
 <img src='./img/Fig9.png' alt='Figure9'><br>
 <figurecaption>Figure 9.  All the 5 poison images.
 </figurecaption><br>
 <br>
 </figure>
 </div>
