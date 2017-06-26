## PyRecommender: recommens inspection strings for you.

It recommends inspection strings of web apps for vulnerability assessment.  
In the current version, it only supports reflective XSS.  
ex) "></iframe><script>alert();</script>, onmousemove=alert``; 

## Description

PyRecommender has two subsystems.  
1) Investigator
2) Recommender

### Investigator

Investigator investigates possibility of vulnerability while crawling target web apps. In the detail, it sends crafted HTTP requests to each query parameters. And, it vectorizes output places of parameter values and escaping type of symbols / script strings for RXSS.

### Recommender

Recommender creates inspection strings using vectorized values by Investigator and recommends its to security engineers. By the way, Recommender has recommendation engine realized by Machine learning (Multilayer perceptron).

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

## Demo

## VS. 

## Requirement

## Usage

## Install

## Contribution

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[tcnksm](https://github.com/tcnksm)
