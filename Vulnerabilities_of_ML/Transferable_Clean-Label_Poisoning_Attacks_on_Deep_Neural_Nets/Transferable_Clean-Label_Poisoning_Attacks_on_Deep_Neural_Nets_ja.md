# Transferable Clean-Label Poisoning Attacks on Deep Neural Nets
[original paper](https://arxiv.org/abs/1905.05897)  

## Abstract
Clean-label poisoning攻撃は、一見すると無害に見える**汚染画像**を学習データに注入し、この学習データで作成されたモデルに**特定の画像を攻撃者の意図したクラスに誤分類**させることができる。本論文では、標的となるモデルの出力、ネットワーク構造、そして学習データにアクセスせずに攻撃を行うことが可能な手法を検討する。 この目標を達成するために、我々は**Polytope攻撃**と呼ぶ新たなpoisoning攻撃を提案する。この攻撃では、標的とするベース画像を特徴空間で囲むように汚染画像が設計される。また、汚染画像の作成中にDropoutを使用することで、この攻撃の汎用性（様々なモデルに対して攻撃が成功する）が向上することを示す。本手法の成功率は**50％**以上であるが、この時の学習データの汚染率は僅か**1％**である。  

## 1. Introduction
Deep Neural Network（以下、DNN）では、モデルの学習とパラメータチューニングのために大規模なデータセットが必要である。このため、多くのモデル作成者は学習データをWebから収集することが多く、これをWebスクレイピングで自動的に行っている。しかし、最近の研究では、このようなデータ収集方法が脆弱性につながる可能性があることが実証されている。特に信頼できないソースからデータを収集すると、作成するモデルがpoisoning攻撃に対して脆弱になり、攻撃者が悪意を持って作成したデータをデータセットに挿入することで、モデルを乗っ取ることが可能となる。  

本論文では、画像分類のタスクに対する**効果的で汎用的**なClean-label poisoning攻撃を検討する。一般的に、poisoning攻撃は学習データを変更することでモデルの推論（予測）動作を制御することを目的としている。従来のEvasion攻撃やバックドア攻撃とは対照的に、我々は攻撃対象のモデルの推論中に学習データが変更されないケースを調査する。  

Clean-label poisoning攻撃は、従来のpoisoning攻撃とは大きな違いがある。それは、学習データのラベル付け作業をユーザが制御する必要はないことである。よって、汚染画像はユーザによって正しくラベル付けされた場合でも、（誤分類を引き起こす）摂動を維持する必要がある。このような攻撃は、攻撃者がWeb上に汚染画像を配置するだけでデータセットを汚染することができ、WebスクレイピングやSNSの利用者、または汚染に気づかないユーザによって汚染画像が収集されるのを待つだけで良い。収集された汚染画像はユーザによって正しくラベル付けされ、学習データに挿入される。更に、標的型のClean-label攻撃は、モデルの分類精度を無作為に低下させるのではなく、特定のクラスのみをターゲットにすることが可能である。つまり、モデル全体の精度は低下しないため、攻撃の検知を困難にすることができる。  

従来のClean-label poisoning攻撃は、攻撃者が攻撃対象モデルに対する完全な知識を持ち、汚染画像を作成する過程でこの知識を使用するという**ホワイトボックス**設定でのみ実証されている。一方、我々は攻撃対象モデルの内部構造が未知（**ブラックボックス**）の状態で汚染画像を作成することを目指す。  

ブラックボックスのClean-label poisoning攻撃は、以下2つの理由で非常に挑戦的なタスクである。  
 
 * 汚染画像を含むデータセットで学習された攻撃対象モデルの決定境界は予測不能である。  
 * 攻撃者は攻撃対象モデルに直接アクセスすることができないため、汚染画像をモデルに依存しないようにする必要がある。  

ここでは、汎用性の高いClean-label poisoning攻撃を提案する。攻撃者は攻撃対象モデルの出力やハイパーパラメータにアクセスできないが、**攻撃対象モデルと同様のデータセットを収集できる**ことを想定している。攻撃者はこのデータセットで**代替モデルを作成**し、**標的空間をConvexの内部に閉じ込めるPolytope**を特徴空間内に形成するNovel Objectiveを最適化する。この汚染画像に過学習するクラスは、標的とする画像を汚染画像と同じクラスに分類する。このNovel Objectiveは、従来のブラックボックス攻撃であるFeature Collision攻撃よりも成功率が高く、ネットワーク内の複数の中間層に適用すると更に攻撃成功率が高まり、転送学習とEnd-to-Endで学習したモデルの両方で高い成功率を示す。また、汚染画像の作成時にDropoutを使用すると、攻撃の汎用性が向上することも示す。  

## 2. The Threat Model
Shafahiらののpoisoning攻撃のように、我々が想定する攻撃者は、少数の汚染画像を攻撃対象モデルが使用する学習データに注入する。攻撃者のゴールは、この学習データで作成された攻撃対象モデルに、テストデータ（当然ながら学習データに含まれない）を特定のクラスに分類させることである。攻撃者が摂動`δ`をベース画像に加えてゴールを達成する画像分類のケースを考えると、汚染画像`x+δ`が正常画像`x`と同じクラスを共有し、特定の方法でDNNの決定境界を変更できるように作成される。  

攻撃対象モデルへの完全なアクセス、またはクエリアクセスを必要とするShafahiらの研究やPapernotらの研究とは異なり、我々は自動運転自動車や監視システムのように、攻撃者が攻撃対象モデルにアクセスできないケースを想定する。その代わりに、攻撃者は攻撃対象モデルの学習データ分布に関する知識を必要とする。これにより、代替モデルの学習用に同様のデータセットを収集できるものとする。  

我々は、攻撃対象モデルが採用する可能性のある2つの学習アプローチを検討する。  

最初の学習アプローチは**転送学習**であり、CIFARやImageNetなどのパブリックなデータセットで学習した**ResNet**のConv層など、（事前学習された後に）Freezeされた特徴抽出器`φ`が画像に適用される。また、重み`W`やバイアス`b`を持つ線形分類器は、別のデータセット`X`の特徴`φ(X)`で微調整される。このタイプの転移学習は、モデル作成時に大規模なラベル付きデータセットが利用できない産業用アプリケーションでは一般的である。 転送学習に対するpoisoning攻撃は、KohとLiangらの研究やShafahiらの研究のように、特徴抽出器の内部構造が明らかになっているホワイトボックス設定で検証されている。  

2番目の学習アプローチは、（特徴抽出と線形分類器が同時に学習される）**End-to-End学習**である。これは明らかに、注入する摂動は特徴抽出器のパラメータ更新に影響を与えるため、このアプローチでは転送学習よりも摂動に対する厳しい制約がある。Shafahiらの10クラス分類設定で約60％の成功率を達成するためにのみ、ターゲットとなるベース画像の最大30％に、約40個の汚染画像に重ね合わせる**透かし戦略**を使用する。  

## 3. Transferable Targeted Poisoning Attacks
### 3.1. Difficulties of Targeted Poisoning Attacks
標的型poisoning攻撃は標的型Evasion攻撃よりも実現困難である。  
画像分類のタスクにおいて、標的型Evasion攻撃は攻撃対象モデルに汚染画像を特定のクラスに誤分類させるのみである。攻撃対象モデルは摂動に適応しないため、攻撃者は制約付き最適化問題を解くことで、決定境界への最短の摂動Pathを見つけるだけで済む。例えば、norm-boundedのホワイトボックス設定では、攻撃者は`δ`を直接最適化するために`δt =  argmin||δ||∞≤ε LCE`をProjected勾配降下法で解き、ターゲットラベル`yt`のクロスエントロピー損失`LCE(xt+δ, yt)`を最小化することができる。クエリアクセスを使用したnorm-boundedのブラックボックス設定では、攻撃者は代替モデルをトレーニングし、蒸留を介して攻撃対象モデルの挙動をシミュレートし、代替モデルに対して同じ最適化を実行して、汎用的な`δ`を見つけることができる。  

一方、標的型poisoning攻撃にはより困難な問題がある。攻撃者は変更されたデータ分布で学習を行った後、ターゲットのベース画像`xt`を代替の標的クラス`y~t`に分類するような攻撃対象モデルを取得する必要がある。 1つの簡単なアプローチは、クラス`y~t`から汚染画像を選択し、特徴空間で汚染画像をできるだけ`xt`に近づけることである。学習時において損失が飽和した後でも汎化は改善し続けることが実際に観察されているため、攻撃対象モデルは通常はデータセットにオーバーフィットする。結果として、汚染画像が特徴空間内の標的画像に近い場合、汚染画像の近くの空間は`y~t`として分類されるため、攻撃対象モデルは`xt`を`y~t`に分類する可能性が高くなる。しかし、図1に示すように、汚染画像と標的画像の距離が短くても攻撃が成功するとは限らない。実際、標的に近づけることは、攻撃を成功させるには制限が多すぎる可能性がある。確かに、汚染画像を標的画像からより遠くに離すことができる条件は存在しますが、その場合攻撃はより成功する。  

 <div align="center">
 <figure>
 <img src='./img/Fig1.png' alt='Figure1'><br>
 <figurecaption>Figure 1. An illustrative toy example of a linear SVM trained on a two-dimensional space with training sets poisoned by Feature Collision Attack and Convex Polytope Attack respectively. The two striped red dots are the poisons injected to the training set, while the striped blue dot is the target, which is not in the training set. All other points are in the training set. Even when the poisons are the closest points to the target, the optimal linear SVM will classify the target correctly in the left figure. The Convex Polytope attack will enforce a small distance of the line segment formed by the two poisons to the target. When the line segment’s distance to the target is minimized, the target’s negative margin in the retrained model is also minimized if it overfits.
 </figurecaption><br>
 <br>
 </figure>
 </div>

### 3.2. Feature Collision Attack
Shafahiらによって提案されたFeature Collision攻撃は、ホワイトボックス設定で汚染画像を生成する手法である。攻撃者は汚染画像`xp`を作成するために、ターゲットのクラスからベース画像`xb`を選択し、`xb`に微小な摂動を追加することにより、`xp`を特徴空間内のターゲット`xt`に近づけることを試みる。具体的には、攻撃者は以下の最適化問題(1)を解いて汚染画像を作成する。  

 <div align="center">
 <figure>
 <img src='./img/Formula1.png' alt='Formula1'><br>
 <br>
 </figure>
 </div>

ここで、`φ`は事前学習された特徴抽出器である。最初の項は、入力空間のベース画像の近くに汚染画像を強制するため、ラベル付けを行うユーザに対して`xb`と同じラベルを維持する。 2番目の項は、汚染画像の特徴表現をターゲットの特徴表現と強制的に衝突させる。ハイパーパラメーター`µ > 0`は、これらの項のバランスのトレードオフを調整する。

汚染画像`xp`がターゲットクラスとして正しくラベル付けされ、攻撃対象モデルのデータセット`X`に注入されている場合、攻撃対象モデルは汚染画像の特徴表現をターゲットクラスに分類するように学習する。次に、`xt`への`xp`の特徴距離が`xp`の間隔よりも小さい場合、`xt`は`xp`と同じクラスに分類され、攻撃は成功する。より攻撃成功率を高めるために、複数の汚染画像が使用される場合もある。

残念ながら、攻撃者にとって異なる特徴抽出器`φ`は異なる特徴空間に繋がっているため、ある特徴空間の距離`d_φ(x_p, x_t) = ||φ(x_p) − φ(x_t)||`が別の特徴空間の距離と一致することを保証するものではない。

しかし、幸運なことにTramerらの研究結果は、各入力画像について、同じデータセットで学習された異なるモデルには敵対部分空間が存在することを示している。このため、適度な摂動を加えることにより、各モデルがサンプルを誤って分類する可能性がある。これは、モデルが同じデータセット、または類似のデータセットで学習されている場合、異なるモデルの特徴空間で汚染画像`xp`をターゲット画像`xt`に近付けることが可能であることを示している。

このような想定の下、ブラックボックス攻撃をシミュレートするための最も明白なアプローチは、汚染画像のセット`{x_p^(j)}_j = 1^k`を最適化し、モデルのアンサンブル` {φ^(i)}_i = 1^m`で特徴衝突を生成することである（`m`はアンサンブルするモデルの数）。このような手法は、ブラックボックス設定のEvasion攻撃でも使用されている。異なる特徴抽出器は、異なる次元・異なる大きさの特徴ベクトルを生成するため、次の正規化された特徴距離を使用する。

 <div align="center">
 <figure>
 <img src='./img/Formula2.png' alt='Formula2'><br>
 <br>
 </figure>
 </div>

### 3.3. Convex Polytope Attack
Feature Collision攻撃の問題点の1つは、画像に摂動を加えると、明らかに不自然な模様が画像に現れることである。クロスエントロピーのような単一エントリの損失を最大化するEvasion攻撃とは異なり、Feature Collision（Shafahiらによる研究）は、汚染画像の特徴ベクトル`φ(xp)`の各エントリを`φ(xt)`に近づけるため、通常、各汚染画像`xp`には数百～数千の制約が生じる。さらに、数式(2)におけるブラックボックス設定では、`m`個のモデルで衝突を強制するため、汚染画像に対する制約の数を更に増やすことになる。このような多数の制約が存在するため、Optimizerは標的の明らかな模様が発生する方向に汚染画像を押し出すため、多くの場合、`xp`を標的クラスのように見せる。その結果、ラベル付けを行うユーザは違和感に気づき、汚染画像をデータセットから除外する可能性が高くなる。図9は、フックの画像から汚染画像を作成し、数式(2)で魚の画像を攻撃する処理過程の例を示している。

 <div align="center">
 <figure>
 <img src='./img/Fig2.png' alt='Figure2'><br>
 <figurecaption>Figure 2. A qualitative example of the difference in poison images generated by Feature Collision (FC) Attack and Convex Polytope (CP) Attack. Both attacks aim to make the model mis-classify the target fish image on the left into a hook. We show two of the five hook images that were used for the attack, along with their perturbations and the poison images here. Both attacks were successful, but unlike FC, which demonstrated strong regularity in the perturbations and obvious fish patterns in the poison images, CP tends to have no obvious pattern in its poisons. More details are provided in the supplementary.
 </figurecaption><br>
 <br>
 </figure>
 </div>

Feature Collision攻撃のもう1つの問題点は汎用性の欠如である。Feature Collision攻撃はブラックボックス設定では失敗する傾向がある。これは、特徴空間内の全てのモデルで汚染画像を`xt`に近づけることが困難であるためである。ニューラルネットワークは1つの線形分類器を使用して特徴`φ(x)`を分離するため、異なる特徴抽出器の特徴空間はかなり異なる。よって、汚染画像`xp`が代替モデルの特徴空間内で`xt`と衝突する場合でも、汎化エラーのために、未知の攻撃対象モデルでは`xt`と衝突しない可能性がある。図1に示すように、`xp`のクラス内サンプルよりも`xt`までの距離が短い場合でも、攻撃は失敗する可能性がある。また、多くの代替モデルをアンサンブルしてこのようなエラーを減らすことは実用的ではない。数式(2)で定義されたアンサンブルなFeature Collision攻撃の実験結果を提供し、効果がないことを示す。

よって、我々は汚染画像で標的の模様がはっきりと表れず、汎化の要件が軽減されるように、汚染画像に対する緩い制約を求める。Shafahiらの研究は通常、複数の汚染画像を使用して1つの標的を攻撃する。先ず、汚染画像の特徴ベクトルのセット`{φ(x_p^(j))}_j = 1^k`の必要十分条件を導き出す。そのため、標的画像`xt`は汚染画像のクラスに分類される。

 * Proposition 1. The following statements are equivalent:  
 <div align="center">
 <figure>
 <img src='./img/Proposition1.png' alt='Proposition1'><br>
 <br>
 </figure>
 </div>

提案1の証明は、巻末の補足資料に記載されている。  
同じクラスの汚染画像セットは、標的画像の特徴ベクトルが汚染画像の特徴ベクトルの凸多面体内にある場合、標的画像のクラスラベルを汚染画像のクラスラベルに変更することが保証される。これは、特徴の衝突を強制するよりもはるかに緩和された制約である。これにより、大きな領域のラベルを変更しながら、汚染画像を標的画像からより遠くに配置することができる。`xt`が未知の攻撃対象モデル内のこの領域内に存在し、`{x_p^(j)}`が期待通りに分類される場合に限り、`xt`は`{x_p^(j)}`と同じラベルとして分類される。

この想定により、標的画像の特徴ベクトルが凸多面体内、または少なくともその近くに位置するように、特徴空間に凸多面体を形成するように汚染画像のセットを最適化する。具体的には以下の最適化問題を解く。

 <div align="center">
 <figure>
 <img src='./img/Formula3.png' alt='Formula3'><br>
 <br>
 </figure>
 </div>

ここで、`x_b^(j)`は`j-th`の汚染画像に対するベース画像であり、`ε`は（人間が知覚できない程度の）摂動の最大許容量である。  

数式(3)は、標的画像が`m`モデルの特徴空間内の汚染画像の凸多面体の中、またはその近くにあるように、汚染画像のセット`{x_p^(j)}`と凸多面体の組み合わせ係数`{c_j^(i)}`のセットを同時に見つける。係数`{c_j ^（i）}`が解かれていることに注意されたい。つまり、モデル毎に異なることが許容されるため、頂点`{x_p^(j)}`を含む凸多面体内の特定の点に近い必要はない。同じ量の摂動が与えられると、数式(2)は数式(3)の特殊なケースであるため、`c_j^(i) = 1/k`を修正するため、そのような目標はFeature Collision攻撃（数式(3)）よりも緩和される。その結果、図9に示すように、汚染画像は殆ど摂動の模様を示さず、Feature Collision攻撃と比較して攻撃のステルス性が向上する。  

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
この節では、全ての画像は`CIFAR10`データセットから取得される。明示的に指定されていない場合、攻撃対象モデルと代替モデル`(φ^(i))`を事前学習するために、学習用データセットの10クラスのそれぞれから最初の4,800画像（合計48,000枚の画像）を取得する。テストデータセットはそのままにし、異なる設定でのこれらのモデルの精度を標準テストセットで評価し、ベンチマークと直接比較できるようにする。攻撃が成功する場合、細工の目立たない画像の摂動だけでなく、ベース画像を基に作成された汚染画像を含むデータセットを微調整した後のテスト精度も変わらないはずである。補足で示すように、攻撃は対応するベース画像を含むデータセットで調整されたモデルの精度と比較し、攻撃対象モデルの精度を維持する。

学習用データセットの残りの2,000枚の画像は、攻撃対象画像の選択、汚染画像の作成、および攻撃対象モデルの微調整のためのバックアップとして使用する。このバックアップ内の各クラスの最初の50個の画像（合計500枚の画像）をクリーンな微調整データセットとして取得する。これは、`Imagenet`のような大規模なデータセットの事前学習済みモデルが、類似しているが通常はバラバラのデータセットで微調整されるシナリオに似ている。攻撃対象のクラスとして「船」を、攻撃対象画像のクラスとして「カエル」をランダムに選択する。つまり、攻撃者は**特定のカエル画像を船として誤分類**させたいと考えるシナリオとする。実験で使用する全ての汚染画像`x_p^(j)`は、500枚の画像の微調整データセットの「船」クラスの最初の5画像から作成される。「カエル」クラスの次の50枚の画像で汚染の有効性を評価する。これらの各画像は、統計を収集するターゲット`xt`として個別に評価される。繰り返すが、攻撃対象画像は学習データセットおよび微調整セットに含まれていない。

### Networks
この論文では、2組の代替モデルを使用する。セット`S1`には、`SENet18`、`ResNet50`、`ResNeXt29-2x64d`、`DPN92`、`MobileNetV2`、および`GoogLeNet`が含まれている。セット`S2`には、`MobileNetV2`と`GoogLeNet`を除く`S1`の全てのモデルが含まれている。`S1`と`S2`は、以下で指定する様々な実験で使用される。ブラックボックスモデルのアーキテクチャとして、`ResNet18`と`DenseNet121`を使用する。汚染画像は、代替モデルで作成される。6つの異なる代替モデルと2つのブラックボックスモデルの攻撃対象モデルに対するpoisoning攻撃を評価する。ただし、各攻撃対象モデルは、代替モデルとは異なるランダムシードで学習される。代替モデルに攻撃対象モデルが表示される場合、グレーボックス設定とする。それ以外は全てブラックボックス設定である。

`GoogLeNet`の各`Inception`ブロックの出力、および他のネットワークでは各`Residual-like`ブロックの畳み込み層の中央にDropout層を追加する。これらのモデルは、パブリックな[リポジトリ](https://github.com/kuangliu/pytorch-cifar)の同じアーキテクチャとハイパーパラメータ（Dropout除く）を使用し、`0、0.2`、`0.25`、`0.3`のDropout率で前述の48,000画像のデータセットで1から学習する。なお、本検証における攻撃対象モデルはDropoutでは学習されない。

### Attacks
同じ5つの汚染「船」画像を使用し、各「カエル」画像を攻撃する。  
代替モデルには、各モデルにて`0.2`、`0.25`、`0.3`のDropout率で学習された3つのモデルを使用する。これにより、`S1`および`S2`のそれぞれ`18`および`12`の代替モデルが得られる。汚染画像を作成する際、代替モデルの訓練時と同じDropout率を使用する。全ての検証実験で、`ε=0.1`を設定する。モデルは学習データセットに似た画像上に小さな勾配を持つように学習されているため、汚染画像の作成には`0.04`の比較的大きな学習率とし、Optimizerには`Adam`を使用する。各実証実験で汚染の摂動に対して4,000回以下のイテレーションを実行する。なお、特に指定がない限り、最後の層のに対してのみ数式(3)を適用する。  

攻撃対象モデルについては、ファインチューニング中にハイパーパラメータを選択し、前述の攻撃対象モデルの仮定を満たす**500画像の学習データセットを過学習させる**。最後の層の線形分類器のみがファインチューニングされる転移学習では、過学習するために`0.1`という大きな学習率とし、Optimizerには`Adam`を使用する。End-to-Endの設定では、`10^-4`の小さな学習率とし、Optimizerには`Adam`を使用して過学習させる。

### 4.1. Comparison with Feature Collision
We first compare the transferability of poisons generated by FC and CP in the transfer learning training context. The results are shown in Figure 5. We use set S1 of substitute architectures. FC never achieves a success rate higher than 0.5, while CP achieves success rates higher or close to 0.5 in most cases. A qualitative example of the poisons crafted by the two approaches is shown in Figure 4.  

 <div align="center">
 <figure>
 <img src='./img/Fig5.png' alt='Figure5'><br>
 <figurecaption>Figure 5. Success rates of FC and CP attacks on various models. Notice the first six entries are the gray-box setting where the models with same architecture but different weights are in the substitute networks, while the last two entries are the black-box setting.
 </figurecaption><br>
 <br>
 </figure>
 </div>

### 4.2. Importance of Training Set
Despite being much more successful than FC, questions remain about how reliable CP will be when we have no knowledge of the victim’s training set. In the last section, we trained the substitute models on the same training set as the victim. In Figure 6 we provide results for when the substitute models’ training sets are similar to (but mostly different from) that of the victim. Such a setting is sometimes more realistic than the setting where no knowledge of the victim’s training set is required, but query access to the victim model is needed (Papernot et al., 2017), since query access is not available for scenarios like surveillance. We use the less ideal S2, which has 12 substitute models from 4 different architectures. Results are evaluated in the transfer learning setting. Even with no data overlap, CP can still transfer to models with very different structure than the substitute models in the black-box setting. In the 0% overlap setting, the poisons transfer better to models with higher capacity like DPN92 and DenseNet121 than to lowcapacity ones like SENet18 and MobileNetV2, probably because high capacity models overfit more to their training set. Overall, we see that CP may remain powerful without access to the training data in the transfer learning setting, as long as the victim’s model has good generalization.  

 <div align="center">
 <figure>
 <img src='./img/Fig6.png' alt='Figure6'><br>
 <figurecaption>Figure 6. Success rates of Convex Polytope Attack, with poisons crafted by substitute models trained on the first 2400 images of each class of CIFAR10. The models corresponding to the three settings are trained with samples indexed from 1201 to 3600, 1 to 4800 and 2401 to 4800, corresponding to the settings of 50%, 50% and 0% training set overlaps respectively.
 </figurecaption><br>
 <br>
 </figure>
 </div>

### 4.3. End-to-End Training
A more challenging setting is when the victim adopts endto-end training. Unlike the transfer learning setting where models with better generalization turn out to be more vulnerable, here good generalization may help such models classify the target correctly despite the poisons. As shown in Figure 7, CP attacks on the last layer’s feature is not enough for transferability, leading to almost zero successful attacks. It is interesting to see that the success rate on ResNet50 is 0, which is an architecture in the substitute models, while the success rate on ResNet18 is the highest, which is not an architecture in the substitute models but should have worse generalization.  

Therefore, we enforce CP in multiple layers of the substitute models, which breaks the models into lower capacitie ones and leads to much better results. In the gray-box setting, all of the attacks achieved more than 0.6 success rates. However, it remains very difficult to transfer to GoogLeNet, which has a more different architecture than the substitute models. It is therefore more difficult to find a direction to make the convex polytope survive end-to-end training.  

 <div align="center">
 <figure>
 <img src='./img/Fig7.png' alt='Figure7'><br>
 <figurecaption>Figure 7. Success rates of Convex Polytope Attack in the end-toend training setting. We use S2 for the substitute models.
 </figurecaption><br>
 <br>
 </figure>
 </div>

## 5. Conclusion
In summary, we have demonstrated an approach to enhance the transferability of clean-labeled targeted poisoning attacks. The main contribution is a new objective which constructs a convex polytope around the target image in feature space, so that a linear classifier which overfits the poisoned dataset is guaranteed to classify the target into the poisons’ class. We provided two practical ways to further improve transferability. First, turn on Dropout while crafting poisons, so that the objective samples from a variety (i.e. ensemble) of networks with different structures. Second, enforce the convex polytope objective in multiple layers, which enables attack success even in end-to-end learning contexts. Additionally, we found that transferability can depend on the data distribution used to train the substitute model.  

## A. Proof of Proposition 1
`2 ⇒ 1`  

For multi-class problems, the condition for `φ(x)` to be classified as `ℓ_p` is  

 <div align="center">
 <figure>
 <img src='./img/A-1.png' alt='FormulaA-1'><br>
 <br>
 </figure>
 </div>

Each of these constraints is linear, and is satisfied by a convex half-space. The region that satisfies all of these constraints in an intersection of convex half-spaces, and so is convex. Under condition (2), `φ(xt)` is a convex combination of points in this convex region, and so `φ(xt)` is itself in this convex region.  

`2 ⇒ 1`  

Suppose that (1) holds. Let  

 <div align="center">
 <figure>
 <img src='./img/A-2.png' alt='FormulaA-2'><br>
 <br>
 </figure>
 </div>

be the convex hull of the points `φ(x_p^j)}_j=1^k` Let `t = argmin_u∈S||u − φ(x_t)||` be the closest point to `φ(x_t)` in `S`. If `||u_t − φ(x_t)|| = 0`, then (2) holds and the proof is complete. If `||u_t − φ(x_t)|| > 0`, then define the classifier function.  

 <div align="center">
 <figure>
 <img src='./img/A-3.png' alt='FormulaA-3'><br>
 <br>
 </figure>
 </div>

Clearly `f(φ(x_t)) < 0`. By condition (1), there is some `j` with `f(φ(x_p^j)) < 0` as well. Consider the function  

 <div align="center">
 <figure>
 <img src='./img/A-4.png' alt='FormulaA-4'><br>
 <br>
 </figure>
 </div>

Because ut is the closest point to `φ(x_t)` in `S`, and `g` is smooth, the derivative of `g` with respect to `η`, evaluated at `η = 0`, is `0`. We can write this derivative condition as  

 <div align="center">
 <figure>
 <img src='./img/A-5.png' alt='FormulaA-5'><br>
 <br>
 </figure>
 </div>

However this statement is a contradiction, since `f(φ(x_p^j)) < 0`.  

## B. Comparison of Validation Accuracies
To make data poisoning attacks undetectable, in addition to making the perturbations to nonobvious, the accuracy of the model fine-tuned on the poisoned dataset shall not drop too significantly, compared with fine-tuning on the same (except for the poisons) clean dataset. Figure 8 shows that the drop in accuracy is indeed not obvious.

## C. Details of the qualitative example
Both the target fish image and the five hook images used for crafting poisons come from the WebVision (Li et al., 2017) dataset, which has the same taxonomy as the ImageNet dataset. Figure 9 gives all the five poison examples.

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
