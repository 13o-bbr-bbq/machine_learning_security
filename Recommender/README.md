## PyRecommender: recommends inspection strings for you.

It recommends inspection strings of web apps for vulnerability assessment.  
In the current version, it only supports reflective XSS.  
ex) "></iframe><script>alert();</script>, onmousemove=alert``; 

[more info](http://www.mbsd.jp/blog/takaesu_index.html)

### system overview
![PyRecommender overview](system_overview.png)

## Description

PyRecommender has two subsystems.  
* Investigator
* Recommender

### Investigator

**Investigator** investigates possibility of vulnerability while crawling target web apps. In the detail, it sends crafted HTTP requests to each query parameters. And, it vectorizes output places of parameter values and escaping type of symbols / script strings for RXSS.

### Recommender

**Recommender** computes recommended inspection strings using vectorized values by Investigator and recommends its to security engineers. By the way, Recommender has recommendation engine realized by **Machine learning (Multilayer perceptron)**.

#### Examples
```
Recommender>python main.py http://192.168.0.6/
Using Theano backend.
Using gpu device 0: GeForce GTX 965M (CNMeM is disabled, cuDNN 5110)

target url: http://192.168.0.6/xss/reflect/onmouseover_unquoted?in=changeme5, parameter: in
feature: [5, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
Loading learned data from recommender.h5
('126', 0.78671795)
('5', 0.18429208)
('53', 0.019127412)
Elapsed time  :2.6320002079[sec]

target url: http://192.168.0.6/xss/reflect/onmouseover_div_unquoted?in=changeme6, parameter: in
feature: [10, 5, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
Loading learned data from recommender.h5
('130', 0.90387237)
('110', 0.074135423)
('138', 0.0083016483)
Elapsed time  :0.283999919891[sec]
```
Above number (130, 110, 126 e.g.,) is corresponding each inspection strings.  
You have to pre-learning PyRecommender using such as [this data set](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Recommender/train_data/train_xss.csv).

### explanatory variable
#### Output plaaes of parameter values
op_html: HTML tag types  
op_attr: Attribute types  
op_js: Output places in JavaScript  
op_vbs:	Output places in VBScript  
op_quot: Quotation types  

#### Escaping types
esc_double: Double quotation (") is Pass (0) or Fail (1).  
esc_single: Single quotation (') is Pass (0) or Fail (1).  
esc_back:	Back quotation (\`) is Pass (0) or Fail (1).  
esc_left: Left symbol (<) is Pass (0) or Fail (1).  
esc_right: Right symbol (>) is Pass (0) or Fail (1).  
esc_alert: Script string (alert();) is Pass (0) or Fail (1).  
esc_prompt: Script string (prompt();) is Pass (0) or Fail (1).  
esc_confirm: Script string (confirm();) is Pass (0) or Fail (1).  
esc_balert: Script string (alert\`\`;) is Pass (0) or Fail (1).  
esc_sscript: Script tag (<script>) is Pass (0) or Fail (1).  
esc_escript: Script tag (</script>) is Pass (0) or Fail (1).  
esc_msgbox: Script tag (Msgbox();) is Pass (0) or Fail (1).  

### response variable
label: label name  
inspection_strings: Inspection strings corresponding each labels.  

***
PyRecommender converts above feature to vector using predefined convertion table.  
Conversion table is [here](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Recommender/temp/convert_table_en.png).

## Usage

### Train
```
Recommender>python main.py TRAIN
```
[Demo](https://www.youtube.com/watch?v=V2sqJIfYiKk)

### Recommend
Recommender>python main.py [target url]  
```
Recommender>python main.py http://192.168.0.6/
```
[Demo](https://www.youtube.com/watch?v=0PlQM1NwXlw)

## Requirement libraries
* pandas
* requests
* beautifulsoup4
* numpy
* scikit-learn
* keras

## Operation check environment
* Python 2.7.12 (Anaconda2)
* pandas 0.18.1
* requests 2.13.0
* beautifulsoup4 4.5.1 
* numpy 1.12.1
* scikit-learn 0.18.1
* keras 1.1.1

## Licence

[Apache License 2.0](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Recommender/LICENSE)

## Author

[Isao Takaesu](https://github.com/13o-bbr-bbq)
