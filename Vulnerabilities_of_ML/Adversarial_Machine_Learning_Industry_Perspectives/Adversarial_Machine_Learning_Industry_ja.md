# Adversarial Machine Learning - Industry Perspectives
[original paper](https://arxiv.org/abs/2002.05646)  

## Abstract
28の組織へのインタビューにより、AIを導入している多くの組織は、機械学習（ML）システムに対する攻撃を防御・検知、および適切にインシデント・レスポンス（IR）するための戦略を備えていないことが明らかになった。我々はインタビューから得られた洞察を活用し、機械学習システムの保護という観点から、従来のソフトウェア・セキュア開発におけるギャップを列挙する。このホワイトペーパーは、MLシステムの設計・開発・リリース時にMLシステムのセキュリティを確保する必要のある「開発者/MLエンジニア」と、インシデントが発生した際の「IR担当者」という2つの観点から書かれている。このホワイトペーパーの目標は、MLシステムに対する攻撃が横行する時代に向け、ML研究者をソフトウェア・セキュア開発ライフサイクルの修正に関与させることである。  

## I Introduction
Adversarial Machine Learning is now having a moment in the software industry - For instance, Google [1], Microsoft [2] and IBM [3] have signaled, separate from their commitment to securing their traditional software systems, initiatives to secure their ML systems. In Feb 2019, Gartner, the leading industry market research firm, published its first report on adversarial machine learning [4] advising that Application leaders must anticipate and prepare to mitigate potential risks of data corruption, model theft, and adversarial samples. The motivation for this paper is to understand the extent to which organizations across different industries are protecting their ML systems from attacks, detecting adversarial manipulation and to responding to attacks on their ML systems.  

MLシステムに対する攻撃は、今やソフトウェア業界では当たり前に認識されている。例えば、Google[1]、Microsoft[2]、IBM[3]は、従来のソフトウェアシステムのセキュア化とは別に、MLシステムの保護への取り組みを行っている。2019年2月、市場調査会社であるGartnerは、MLシステムを開発するリーディングカンパニーが、データ汚染、モデル抽出（窃盗）、およびAdversarial Examplesの潜在的なリスクを予測し、これらに対応する必要があることを示した世界初のレポート[4]を公開した。このホワイトペーパーの目的は、様々な業界の組織が攻撃からMLシステムを保護し、不正行為をいち早く検出すること重要性を理解させることである。  

組織がMLシステムを体系的なセキュリティで保護する理由は多く存在する。先ず、過去3年間に機械学習に多大な投資をしている企業「Google、Amazon、Microsoft、Tesla」は、規模の大きなMLシステムに対する攻撃に直面した[5]～[8]。次に、ISO[9]やNIST[10]のような標準化団体は、MLシステムのセキュリティを評価するための認証基準を作成しており、その基準は業界で求められている[11]。また、政府は業界がMLシステムを安全に構築するためのルール策定に乗り出しており、EUはMLシステムの信頼性を評価するためのチェックリストをリリースした。これにより、MLシステムの開発は規制に"がんじがらめ"になる可能性がある。なお、2020年の機械学習への投資の年間成長率は39％と予測されているため[13]、組織がMLシステムの保護に投資するのは当然のことである。  

このホワイトペーパーでは、以下2つの貢献をしている:  

 1. MLシステムを保護する必要性があるにも関わらず、28もの組織に跨る調査において、殆どの組織はMLシステムの保護に対応していないことが判明した。28の組織のうち25が、MLシステムを保護するための適切な戦略を持っていないことも判明した。  

 2.セキュア開発ライフサイクル（SDL）フレームワークを使用し、MLシステムを構築するセキュア開発の一例を列挙する。これは、業界における事実上のソフトウェア構築プロセスである。  

このホワイトペーパーは、一般的なソフトウェア開発会社が直面するMLシステムの保護における問題点とギャップの概要である。機械学習システムを保護する際に、ソフトウェア開発者/MLエンジニアとIR担当者という2つの役割が直面する問題を解決するために、研究者のコミュニティに訴えたいと考えている。このホワイトペーパーの目標は、MLの研究者が、MLシステムに対する攻撃が横行する時代に向けて、ML研究者をソフトウェア・セキュア開発ライフサイクルの修正に関与させることである。  

このホワイトペーパーは2つのパートから構成される。最初のパートでは、我々の調査方法と調査結果の概要を説明する。2番目のパートは、MLシステムを設計・実装する際に、3つのフェーズでMLシステムを保護する際のギャップを説明する。  

# II INDUSTRY SURVEY ABOUT ADVERSARIAL ML
我々は、Fortune 500、中小企業、非営利団体、政府機関に跨る28の組織にインタビューし、攻撃からMLシステムを保護する方法を調査した（調査対象組織の概要は表Iおよび表IIを参照のこと）。  

我々が調査した28組織の内、22の組織は金融、コンサルティング、サイバーセキュリティ、ヘルスケア、政府などの機微な情報を保持する分野に属していた。残りの6組織は、ソーシャルメディア分析、出版、農業、都市計画、食品加工、翻訳サービスとなっていた（調査組織の分布は表IIを参照のこと）。  

**TABLE I. Organization size**  
|rganization size|Cunt|
|:---:|:---:|
|Large Organizations (> 1,000 employees)|18|
|Small-and-Medium size Businesses|10|  

**TABLE II. Organization types**  
|Organization|Count|
|:---:|:---:|
|Cybersecurity|10|
|Healthcare|5|
|Government|4|
|Consulting|2|
|Banking|2|
|Social Media Analytics|1|
|Publishing|1|
|Agriculture|1|
|Urban Planning|1|
|Food Processing|1|
|Translation|1|

各組織の調査では、「MLシステムを開発する開発者」と「セキュリティ担当者」にインタビューを行った。組織の規模に応じて、これらの2つの役割は、異なるチーム、同じチーム、または兼任であった。我々がインタビューした全ての組織は、従来のソフトウェア開発におけるセキュア開発ライフサイクルに精通していたが、その実践の程度は、中小企業よりも文書化されたセキュア開発プロセスを持つ大企業の方が高かった。また、機械学習への投資が比較的成熟し、AIを核にしたビジネスを展開する企業もあった。  

我々が調査した組織は様々な方法でMLシステムを実装している。殆どは、Keras・TensorFlow・PyTorchなどのMLフレームワークを使用していた。10の組織は、Microsoft Cognitive API[14]、Amazon AIサービス[15]、Google CloudAI[16]などのサービスに依存していた。なお、MLシステムをフルスクラッチで開発している組織は2つだけであった（表IIIを参照のこと）。  

**TABLE III. ML STRATEGY**  
|How do you build ML Systems|Count|
|:---:|:---:|
|Using ML Frameworks|16|
|Using ML as a Service|10|
|Building ML Systems from scratch|2|

### Limitations of Study:
我々が調査した28の組織は、機械学習を利用している組織全体を表しているとは限らない。例えば、今回の調査には新興企業は含まれておらず、調査にはセキュリティに注意を払う組織が圧倒的に多いという特徴がある。  

また、殆どの組織が米国またはヨーロッパで事業を展開しており、本社を構えている地理的分布も考慮していない。また、調査は攻撃者によって引き起こされる障害に限定し、一般的な破損[17]、報酬ハッキング[18]、分布変化[19]、または、自然発生するAdversarial Examples[20]などは調査に含めていない。  

### A. Findings:
 1. 28の組織全てが、AIシステムのセキュリティがビジネスにとって重要であるとの認識を示したが、依然として従来のセキュリティに重点が置かれている。あるセキュリティアナリストが言うように、我々の最大の脅威ベクターは、スピアフィッシングとマルウェアである。MLシステムに対する攻撃は未だ現実のものではない。MLシステムへの攻撃には大きな関心が寄せられているが、この問題に対処する人員を割り当てる準備ができている組織は6つだけである（大規模組織または政府）。  

 2. 機械学習セキュリティのノウハウ欠如：組織はMLシステムを保護するための知識を欠いているように思われる。調査対象組織のある1人が言うように、従来のソフトウェア攻撃は既知であるが、MLシステムへの攻撃は未知のものである。25組織中22組織（政府機関の3組織がこの質問に対する満足な回答は無かった）は、MLシステムを保護するための適切なツールが用意されていないと述べており、明示的なガイダンスを探しているとのことであった。また、殆どのセキュリティエンジニアは、MLシステムに対する攻撃を検知して対応することができない（表IVを参照のこと）。  

**TABLE IV. STATE OF ADVERSARIAL ML**  
|Do you secure your ML systems today|Count|
|:---:|:---:|
|Yes|3|
|No|22|

 3. [21]に概説されている攻撃の一覧を調べ、ビジネスに影響を与える可能性のある攻撃を選択するよう依頼した（表Vを参照のこと）。注：回答者は全ての攻撃をランク付けするのではなく、1つの攻撃のみを選択した。結果は次のとおりである。  

**TABLE V. TOP ATTACK**  
|Which attack would affect your org the most?|Distribution|
|:---:|:---:|
|Poisoning (e.g: [22])|10|
|Model Stealing (e.g: [23])|6|
|Model Inversion (e.g: [24])|4|
|Backdoored ML (e.g: [25])|4|
|Membership Inference (e.g: [26])|3|
|Adversarial Examples (e.g: [27])|2|
|Reprogramming ML System (e.g: [28])|0|
|Adversarial Example in Physical Domain (e.g: [5])|0|
|Malicious ML provider recovering training data (e.g: [29])|0|
|Attacking the ML supply chain (e.g: [25])|0|
|Exploit Software Dependencies (e.g: [30])|0|

 * データ汚染は多くの組織の注目を集めた。中規模のフィンテック企業はこのように述べている。「我々はMLシステムを使用して顧客に金融商品を提案している。 MLシステムの整合性は非常に重要である。Microsoftのチャットボット「Tay（偏った学習データを学習したことで、差別的な発言を行った）」のように、不適切なことを顧客に発言しないか心配である。」。  

 * 組織は、プライバシー侵害に繋がる可能性のある攻撃も重視している。ある銀行の担当者は「顧客情報やMLモデルで使用される従業員情報を保護したいと考えているが、未だ計画は整っていない。」。  

 * 知的財産の損失に繋がる可能性のあるモデル抽出（窃盗）も懸念されている。大規模な小売組織は独自のアルゴリズムでMLシステムを運用しているため、攻撃者にMLシステム内のモデルをリバースエンジニアリングされないか懸念している。  

 * 物理ドメインにおけるAdversarial Examplesに回答者は興味を示したが、リストの上位にはランクされなかった。理由の一つは、我々がインタビューした組織は、自動運転自動車やドローンのような物理的な要素を持っていなかったかもしれない。  

 4. セキュリティアナリストにとってMLシステムに対する攻撃は関心事項であるが、現実問題とはギャップがあると感じている。多くのセキュリティアナリストは、Keras、TensorFlow、PyTorchなどのMLフレームワークで利用可能なアルゴリズムはセキュアに実装されており、セキュリティテストも実施済みであることを期待している。 これはおそらく、FacebookやGoogleなどの大規模な組織によって公開されたライブラリは、既にセキュリティテストが実施されていると思い込んでいるからである。同様に、Machine Learning as a Serviceを使用している組織の回答者が言ったように、組織はMLシステムのセキュリティ責任をサービスプロバイダーに持たせようとしているようである。  

 5. 最後に、セキュリティアナリストと開発者は、MLシステムが攻撃された場合の影響を理解していない。あるMLエンジニアの一人が言ったように、「私はどのMLシステムもスプーフィングの影響を受けないとは思っていないが、潜在的な障害モードと同様に、信頼レベルと期待されるパフォーマンスを知る必要がある。MLシステムがスプーフィングされた場合の最悪の影響は何でしょうか？」。  

Figure.1 に示す要約は、MLシステムを構築する際の現在のSDLプロセスのギャップについて詳しく説明している。各ギャップについて、従来のソフトウェア開発における既存手法の概要を示し、将来の研究課題を描いている。  

 <div align="center">
 <figure>
 <img src='./img/Fig1.png' alt='Figure1'><br>
 <br>
 </figure>
 </div>

## III ABOUT SDL
2001年7月、Microsoft Internet Information Server（IIS）4.0および5.0[31]はコンピューターワーム「CodeRed」の攻撃を受けた。これは、IIS4およびIIS5システムでデフォルトで実行されるコードのバグが原因で発生し、バッファオーバーフロー攻撃が可能となった。2002年1月、Microsoft はシステムに存在する既知の脆弱性を全て修正するために、2か月間に渡り新しいソフトウェア開発を停止し、セキュリティ専門家と開発者で対応を行った。この緊密な相互作用により、セキュリティガイダンスを提供する体系的なプロセスが進化し、エンジニアがソフトウェアのバグや実装のバグを探し出すのに大いに役立った。この一連の経験は、セキュア開発ライフサイクル（SDL）と呼ばれるようになった。SDLは全てのソフトウェアのバグを排除できる訳ではないが、脆弱性を含んだソフトウェアが顧客の手に届く前に、それを発見するのに役立つ。例えば、MicrosoftにSDLが導入された後、Windows XP～Windows Vistaで報告された脆弱性数は45％減少し、SQL Server 2000～SQL Server 2005で報告された脆弱性数は91％減少した[32]。現在、SDLは何らかの形でGoogle[34]、IBM[35]、Facebook[36]、Netflix[37]を含む122の組織[33]で採用されており、業界のデファクトスタンダードなソフトウェア開発プロセスとなっている。  

我々の主な調査は、従来のソフトウェアの保護に使用されるSDLプロセスの修正と改訂であり、MLシステムを攻撃から保護する。  

## IV GAPS DURING DEVELOPMENT OF ML SOLUTION
### A. Curated repository of attacks
In traditional software security, attacks are decomposed into shareable tactics and procedures and are collectively organized in the MITRE ATT&CK framework [38]. This provides a search-able attack repository comprising, attacks by researchers as well as nation state attackers. For every attack, there is a description of the technique, which advanced persistent threat is known to use it, detection ideas as well as reference to publications with further context.  

In adversarial ML, the scholarship is booming [39] but the awareness is low among developers and security analysts only 5 out of 28 organizations stated that they had working knowledge of adversarial ML. We propose that a similar curated repository of attacks be created, preferably by extending the widely used existing MITRE Framework. For instance, when adversarial ML researchers publish a new type of attack, we ask them to register their attacks in the MITRE framework, so that security analysts have a unified view of traditional and adversarial ML attacks.  

### B. Adversarial ML specific secure coding practices:
In traditional software setting, secure coding practice enables engineers to reduce exploitable vulnerabilities in their programs and enables auditing of source code by other engineers. For instance Python [40], Java, C and C++ [41] have well defined secure coding practice against traditional software bugs like memory corruption. In machine learning setting, adversarial ML specific security guidance is sparse. Most toolkits provide best practices (TensorFlow [42] , Pytorch [43], Keras [44]) but TensorFlow is the only framework that provides consolidated guidance around traditional software attacks [45] and links to tools for testing against adversarial attacks [46].  

We think future work in adversarial ML should focus on providing best practices to eliminate undefined program behaviors and exploitable vulnerabilities. We acknowledge it is difficult to provide concrete guidance because the field is protean [47]. Perhaps one direction, would be to enumerate guidance based on security consequence. Viewing the world through SDL allows for imperfect solutions to exist. For instance, in traditional software security, the outdated cryptgenradnom [48] function should not be used to generate random seeds for secret sharing protocols which are of higher security consequence), but can be used to generate process IDs in an operating system (which is of lower security consequence). Instead of thinking of secure coding practice as underwriting a strong security guarantee, a good start would be to provide examples of security-compliant and non-compliant code examples.  

### C. Static Analysis and Dynamic Analysis of ML Systems
In traditional software security, static analysis tools help detect potential bugs in the code without the need for execution and to detect violations in coding practices. The source code is generally converted into an abstract syntax tree, which is then used to create a control flow graph. Coding practices and checks, which are turned into logic, are searched over the control flow graph, and are raised as errors when inconsitent with logic. In traditional software for instance, in Python tools like Pyt [49] detect traditional software security vulnerabilities. Dynamic analysis, on the other hand, involves searching for vulnerabilities on executing a certain code path.  

On the ML front, tools like cleverhans [46], secml [50], and the adversarial robustness toolkit [51] providing a certain degree of white-box style and blackbox style dynamic testing. A future area of research is how to extend the analysis to model stealing, model inversion and membership inference style attacks. Out of the box static analysis for adversarial ML is less explored. One promising angle is work like Code2graph [52] that generates call graphs in ML platform and in conjunction with symbolic execution may provide the first step towards a static analysis tool. We hope that the static analysis tools ultimately integrate with IDE (integrated development environment) to provide analytical insight into the syntax, semantics, so as to prevent the introduction of security vulnerabilities before the application code is committed to the code repository.  

### D. Auditing and Logging in ML Systems
To use a traditional software example, important security events in the operating system like process creation are logged in the host, which is then forwarded to Security Information and Event Management (SIEM) systems. This later enables, security responders to run anomaly detection [53], [54] to detect if an anomalous process (which is an indication of malware) was executed on the machine.  

Auditing in ML systems was initially pointed by Papernot [55] with solution sketches to instrument ML environments to capture telemetry. As done in traditional software security, we recommend that developers of ML systems, identify high impact activities in their system. We recommend executing the list of attacks that are considered harmful to the organization and ensuring that the events manifesting in the telemetry can be traced back to the attack. Finally, these events must be exportable to traditional Security Information and Event Management systems, so that analysts can keep an audit trail for future investigations.  

### E. Detection and Monitoring of ML systems
Currently, ML environments are illegible to security analysts as they have no operational insights. There has been insightful working pointing to the brittleness of current adversarial detection mechanisms [47] and how to make them better [20]. In addition, we propose that detection methods are written so that they are easily shared among security analysts. For instance, in traditional software security, detection logic is written in a common format, the most popular of which is Sigma [56]. Where MITRE ATT&CK provides a great repository of insight in techniques used by adversaries, Sigma can turn one analyst’s insights into defensive action for many, by providing a way to self-documented concrete logic for detecting an attacker’s techniques.  

## V GAPS WHEN PREPARING FOR DEPLOYMENT OF ML
### A. Automating Tools in Deployment Pipeline
In a typical traditional software setting, after a developer as the developer completes small chunks of the assigned task, the following sequence of steps generally follow: first, the code is committed to source control and Continuous Integration triggers application build and unit tests; once these are passed, Continuous Deployment triggers an automated deployment into testing and then production wherein it reaches the customer. At each step of the build, security tools are integrated.  

We hope that dynamic analysis tools built for adversarial ML are integrated into the continuous integration / continuous delivery pipeline. Automating the adversarial ML testing, will help fix issues and without overloading engineers with too many tools or alien processes outside of their everyday engineering experience.  

### B. Red Teaming ML Systems
Informally, the risk of an attack to an organization depends on two factors: the impact it has on the business and the likelihood of the attack occurring. Threat modeling of ML Systems [57], performed by the ML developers, address the impact factor. Red teaming, the deliberate process of exploiting the system through any means possible conducted by an independent security team, helps to assess the likelihood factor. For critical security applications, red teaming is industry standard and a requirement for providing software to US governments [58]. With Facebook being the first industry to start an AI Red Team [59] and is unexplored area in the adversarial ML field for others.  

### C. Transparency Centers
In traditional security, large organizations such as Microsoft [60], Kaspersky [61], Huawei [62] have provided transparency centers where participants visit a secure facility to conduct deep levels of source code inspection and analysis. Participants would have access to source code and an environment for indepth inspection with diagnostic tools to verify the security aspects of different products such as SSL and TCP/IP implementation or pseudorandom number generators.  

In adversarial ML context, future transparency centers may need to attest over 3 modalities: that the ML platform is implemented in a secure fashion; that the MLaaS is implemented meeting basic security objectives and finally, that the ML model embedded in an edge device (such as models on mobile phones, for instance) meets basic security objectives. An interesting direction for future research is to providing tools/test harnesses to advance security assurance of products building on top of formal verification such as [63], [64] to extend to large scale ML models used in industry.  

## VI GAPS WHEN AN ML SYSTEM IS UNDER ATTACK
### A. Tracking and Scoring ML Vulnerabilities
In traditional software security, when a researcher finds a vulnerability in a system, it is first assigned a unique identification number and registered in a database called Common Vulnerabilities and Exposure [65]. Accompanying these vulnerabilities are severity ratings calculated by using Common Vulnerability Scoring System [66]. For instance, in the recent zero day found against Internet Explorer that allowed for remote code execution [67] the vulnerability was referred to as ”CVE-2020-0674” and had assigned a base CVSS score 7.5 out of 10 [68], roughly indicating the seriousness of the bug. This enables the entire industry to refer to the problem using the same tongue.  

In an ML context, we ask the adversarial ML research community to register vulnerabilities (especially affecting large groups of consumers) in a trackable system like CVE to ensure that industry manufacturers are alerted. It is not clear how ML vulnerabilities should be scored accounting for risk and impact. Finally, When a security analyst sees news about an attack, the bottom line is mostly Is my organization affected by the attack? and today, organizations lack the ability to scan an ML environment for known adversarial ML specific vulnerabilities.  

### B. Incident Response
When a security engineer receives a notification that an ML system is under attack, and triages that the attack is relevant to the business, there are two important steps ascertaining blast radius and preparing for containment. For instance, in the case of ransomware, a traditional software attack, the blast radius would be to determine other machines connected to the infected machine, and containment would be to remove the machines from the network for forensic analysis.  

Both steps are difficult, because ML systems are highly integrated in a production setting where a failure of one can lead to unintended consequences [69]. One interesting line of research is to identify whether, if it is possible to container-ize ML systems so as to quarantine uncompromised ML systems from the impact of a compromised ML system, just as anti virus systems would quarantine an infected file.  

### C. Forensics
In traditional software security, once the machine is contained, it is prepared for forensics to ascertain root cause. There are a lot of open questions in this area so as to meaningfully interrogate ML systems under attack to ascertain the root cause of failure:  

 1. What are the artifacts that should be analyzed for every ML attack? Model file? The queries that were scored? Training data? Architecture? Telemetry? Hardware? All the software applications running on the attacked system? How can we leverage work data provenance and model provenance for forensics?  

 2. How should these artifacts be collected? For instance, for ML models developed on the end point or Internet of Things vs. organizations using ML as a Service, the artifacts available for analysis and acquisition methodology will be different. We posit that ML forensics methodology is dependent on ML frameworks (like PyTorch vs. TensorFlow), ML paradigms (e.g: reinforcement learning vs. supervised learning) and ML environment. (running on host vs cloud vs edge).  

 3. An orthogonal step that may be carried out is cyberthreat attribution, wherein the security analyst is able to determine the actor responsible for the attack. In traditional software, this is done by analyzing the forensic evidence such as infrastructure used to mount the attack, threat intelligence and ascertaining the attackers tools, tactics and procedures using established rubrics called analytic trade craft [70]. It is unclear how this would be amended in the adversarial ML age.  

### D. Remediation
In traditional software security. Tuesday is often synonymous with Patch Tuesday. This is when companies like Microsoft, SAS, and Adobe release patches for vulnerabilities in their software, which are then installed based on an organization’s patching policy.  

In an ML context, when Tay was compromised because of poisoning attack, it was suspended by Microsoft. This may not be possible for all ML systems, especially those that have been deployed on the edge. It is not clear what the guidelines are for patching a system, that is vulnerable to model . On the same lines, it is not clear how one would validate if the patched ML model will perform as well as the previous one, but not be subject to the same vulnerabilities based on Papernot et. als [71] transferability result.

## VII CONCLUSION
In a keynote in 2019, Nicholas Carlini [72] likened the adversarial ML field to crypto pre-Shannon based on the ease with which defenses are broken. We extend Carlinis metaphor beyond just attacks and defenses: through a study over 28 organizations, we conclude that most ML engineers and security incident responders are unequipped to secure industry-grade ML systems against adversarial attacks. We also enumerate how researchers can contribute to Security Development Lifecyle (SDL), the de facto process for building reliable software, in the era of adversarial ML. We conclude similar that if Machine learning is Software is 2.0 [73], it also needs to inherit the structured process of developing about security from traditional software 1.0 development process.  

## REFERENCES
[1] “Responsible AI Practices.” [Online]. Available: https://ai.google/responsibilities/responsible-ai-practices/?category=security  

[2] “Securing the Future of AI and ML at Microsoft.” [Online]. Available: https://docs.microsoft.com/en-us/security/securing-artificial-intelligence-machine-learning  

[3] “Adversarial Machine Learning,” Jul 2016. [Online]. Available: https://ibm.co/36fhajg  

[4] S. A. Gartner Inc, “Anticipate Data Manipulation Security Risks to AI Pipelines.” [Online]. Available: https://www.gartner.com/doc/3899783  

[5] A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, “Synthesizing robust adversarial examples,” arXiv preprint arXiv:1707.07397, 2017.

[6] J. Li, S. Qu, X. Li, J. Szurley, J. Z. Kolter, and F. Metze, “Adversarial Music: Real World Audio Adversary Against Wake-word Detection System,” in Advances in Neural Information Processing Systems, 2019, pp. 11 908–11 918.  

[7] P. L. Microsoft, “Learning from Tay’s introduction,” Mar 2016. [Online]. Available: https://blogs.microsoft.com/blog/2016/03/25/learning-tays-introduction/  

[8] “Experimental Security Research of Tesla Autopilot,” Tech. Rep. [Online]. Available: https://bit.ly/37oGdla  

[9] “ISO/IEC JTC 1/SC 42 Artificial Intelligence,” Jan 2019. [Online]. Available: https://www.iso.org/committee/6794475.html  

[10] “AI Standards.” [Online]. Available: https://www.nist.gov/topics/artificial-intelligence/ai-standards,  

[11] R. Von Solms, “Information security management: why standards are important,” Information Management & Computer Security, vol. 7, no. 1, pp. 50–58, 1999.  

[12] “Ethics guidelines for trustworthy ai,” Nov 2019. [Online]. Available: https://ec.europa.eu/digital-single-market/en/news/ethics-guidelines-trustworthy-ai  

[13] “2018 AI predictions 8 insights to shape business strategy,” Tech. Rep. [Online]. Available: https://www.pwc.com/us/en/advisory-services/assets/ai-predictions-2018-report.pdf  

[14] [Online]. Available: https://azure.microsoft.com/en-us/services/cognitive-services/  

[15] [Online]. Available: https://aws.amazon.com/machine-learning/ai-services/  

[16] [Online]. Available: https://cloud.google.com/products/ai/  

[17] D. Hendrycks and T. Dietterich, “Benchmarking neural network robustness to common corruptions and perturbations,” arXiv preprint arXiv:1903.12261, 2019.  

[18] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mane, “Concrete problems in ai safety,” ´ arXiv preprint arXiv:1606.06565, 2016.  

[19] J. Leike, M. Martic, V. Krakovna, P. A. Ortega, T. Everitt, A. Lefrancq, L. Orseau, and S. Legg, “Ai safety gridworlds,” arXiv preprint arXiv:1711.09883, 2017.  

[20] J. Gilmer, R. P. Adams, I. Goodfellow, D. Andersen, and G. E. Dahl, “Motivating the rules of the game for adversarial example research,” arXiv preprint arXiv:1807.06732, 2018.  

[21] R. S. S. Kumar, D. O. Brien, K. Albert, S. Viljoen, and J. Snover, “Failure modes in machine learning systems,” arXiv preprint arXiv:1911.11034, 2019.  

[22] M. Jagielski, A. Oprea, B. Biggio, C. Liu, C. Nita-Rotaru, and B. Li, “Manipulating machine learning: Poisoning attacks and countermeasures for regression learning,” in 2018 IEEE Symposium on Security and Privacy (SP). IEEE, 2018, pp. 19–35.  

[23] F. Tramer, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart, “Stealing machine learning models via prediction apis,” in 25th {USENIX} Security Symposium ({USENIX} Security 16), 2016, pp. 601–618.  

[24] M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks that exploit confidence information and basic countermeasures,” in Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, 2015, pp. 1322–1333.  

[25] T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnerabilities in the machine learning model supply chain,” arXiv preprint arXiv:1708.06733, 2017.  

[26] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in 2017 IEEE Symposium on Security and Privacy (SP). IEEE, 2017, pp. 3–18.  

[27] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” arXiv preprint arXiv:1412.6572, 2014.  

[28] G. F. Elsayed, I. Goodfellow, and J. Sohl-Dickstein, “Adversarial reprogramming of neural networks,” arXiv preprint arXiv:1806.11146, 2018.  

[29] M. Sharif, S. Bhagavatula, L. Bauer, and M. K. Reiter, “Adversarial generative nets: Neural network attacks on state-of-the-art face recognition,” arXiv preprint arXiv:1801.00349, pp. 1556–6013, 2017.  

[30] Q. Xiao, K. Li, D. Zhang, and W. Xu, “Security risks in deep learning implementations,” in 2018 IEEE Security and Privacy Workshops (SPW). IEEE, 2018, pp. 123–128.  

[31] C. C. Zou, W. Gong, and D. Towsley, “Code red worm propagation modeling and analysis,” in Proceedings of the 9th ACM conference on Computer and communications security. ACM, 2002, pp. 138–147.  

[32] [Online]. Available: https://bit.ly/2G4NaMv  

[33] [Online]. Available: https://www.bsimm.com/  

[34] [Online]. Available: https://cloud.google.com/security/overview/whitepaper  

[35] [Online]. Available: https://www.ibm.com/security/secure-engineering/  

[36] [Online]. Available: https://about.fb.com/news/2019/01/designing-security-for-billions/  

[37] [Online]. Available: https://medium.com/@NetflixTechBlog/scaling-appsec-at-netflix-6a13d7ab6043  

[38] [Online]. Available: https://about.fb.com/news/2019/01/designing-security-for-billions/  

[39] N. Carlini. [Online]. Available: https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html  

[40] [Online]. Available: http://www.pythonsecurity.org/  

[41] [Online]. Available: https://wiki.sei.cmu.edu/confluence/display/seccode  

[42] [Online]. Available: https://bit.ly/2RDl3cm  

[43] [Online]. Available: https://pytorch.org/docs/stable/notes/multiprocessing.html  

[44] [Online]. Available: https://keras.io/why-use-keras/  

[45] [Online]. Available: https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md  

[46] N. Papernot, F. Faghri, N. Carlini, I. Goodfellow, R. Feinman, A. Kurakin, C. Xie, Y. Sharma, T. Brown, A. Roy, A. Matyasko, V. Behzadan, K. Hambardzumyan, Z. Zhang, Y.-L. Juang, Z. Li, R. Sheatsley, A. Garg, J. Uesato, W. Gierke, Y. Dong, D. Berthelot, P. Hendricks, J. Rauber, and R. Long, “Technical report on the cleverhans v2.1.0 adversarial examples library,” arXiv preprint arXiv:1610.00768, 2018.  

[47] N. Carlini and D. Wagner, “Adversarial examples are not easily detected: Bypassing ten detection methods,” in Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security. ACM, 2017, pp. 3–14.  

[48] [Online]. Available: https://docs.microsoft.com/en-us/windows/win32/api/wincrypt/nf-wincrypt-cryptgenrandom  

[49] [Online]. Available: https://github.com/python-security/pyt  

[50] M. Melis, A. Demontis, M. Pintor, A. Sotgiu, and B. Biggio, “secml: A Python Library for Secure and Explainable Machine Learning,” arXiv preprint arXiv:1912.10013, 2019.  

[51] M.-I. Nicolae, M. Sinn, M. N. Tran, B. Buesser, A. Rawat, M. Wistuba, V. Zantedeschi, N. Baracaldo, B. Chen, H. Ludwig, I. Molloy, and B. Edwards, “Adversarial Robustness Toolbox v1.1.0,” CoRR, vol. 1807.01069, 2018. [Online]. Available: https://arxiv.org/pdf/1807.01069  

[52] G. Gharibi, R. Tripathi, and Y. Lee, “Code2graph automatic generation of static call graphs for python source code,” in Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering. ACM, 2018, pp. 880–883.  

[53] J. Twycross, U. Aickelin, and A. Whitbrook, “Detecting anomalous process behaviour using second generation artificial immune systems,” arXiv preprint arXiv:1006.3654, 2010.  

[54] W. M. Van der Aalst and A. K. A. de Medeiros, “Process mining and security: Detecting anomalous process executions and checking process conformance,” Electronic Notes in Theoretical Computer Science, vol. 121, pp. 3–21, 2005.  

[55] N. Papernot, “A marauder’s map of security and privacy in machine learning,” arXiv preprint arXiv:1811.01134, 2018.  

[56] F. Roth, “Sigma.” [Online]. Available: https://github.com/Neo23x0/sigma  

[57] N. Papernot, P. McDaniel, A. Sinha, and M. Wellman, “Towards the science of security and privacy in machine learning,” arXiv preprint arXiv:1611.03814, 2016.  

[58] “Nvd.” [Online]. Available: https://nvd.nist.gov/800-53/Rev4/control/CA-8  

[59] B. Dolhansky, R. Howes, B. Pflaum, N. Baram, and C. C. Ferrer, “The deepfake detection challenge (dfdc) preview dataset,” arXiv preprint arXiv:1910.08854, 2019.  

[60] [Online]. Available: https://docs.microsoft.com/en-us/security/gsp/contenttransparencycenters  

[61] [Online]. Available: https://bit.ly/2v89frf  

[62] [Online]. Available: https://www.huawei.com/en/about-huawei/trust-center/transparency/huawei-cyber-security-transparency-centre-brochure  

[63] G. Katz, C. Barrett, D. L. Dill, K. Julian, and M. J. Kochenderfer, “Reluplex: An efficient Smt solver for verifying deep neural networks,” in International Conference on Computer Aided Verification. Springer, 2017, pp. 97–117.  

[64] T.-W. Weng, H. Zhang, H. Chen, Z. Song, C.-J. Hsieh, D. Boning, I. S. Dhillon, and L. Daniel, “Towards fast computation of certified robustness for relu networks,” arXiv preprint arXiv:1804.09699, 2018.  

[65] “Common Vulnerabilities and Exposures (CVE).” [Online]. Available: https://cve.mitre.org/  

[66] “Common Vulnerability Scoring System (CVSS).” [Online]. Available: https://www.first.org/cvss/specification-document  

[67] [Online]. Available: https://portal.msrc.microsoft.com/en-us/security-guidance/advisory/ADV200001  

[68] “CVE-2020-0674.” [Online]. Available: https://kb.cert.org/vuls/id/338824/  

[69] D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, and M. Young, “Machine Learning: The High interest Credit Card of Technical Debt,” in SE4ML: Software Engineering for Machine Learning (NIPS 2014 Workshop), 2014.  

[70] “A Guide to Cyber Attribution,” 2018. [Online]. Available: https://bit.ly/2G50UXB  

[71] N. Papernot, P. McDaniel, and I. Goodfellow, “Transferability in machine learning: from phenomena to black-box attacks using adversarial samples,” arXiv preprint arXiv:1605.07277, 2016.  

[72] [Online]. Available: https://youtu.be/-p2il-V-0fk?t=1574  

[73] “Software 2.0,” 2017. [Online]. Available: https://medium.com/@karpathy/software-2-0-a64152b37c35  
