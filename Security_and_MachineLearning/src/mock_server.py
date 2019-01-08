#!/bin/env python
# -*- coding: utf-8 -*-
import time
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
define('port', default=8888, help='run on the given port.', type=int)


class WebHandler(tornado.web.RequestHandler):
    base_time = 0.0

    def get(self):
        param = self.get_argument('param')
        # Normal response time.
        if param == 'normal':
            WebHandler.base_time = 0.01
        # Heavy response time.
        if param == 'attack':
            WebHandler.base_time = 0.5
        # Increasing response time.
        if param == 'load':
            WebHandler.base_time += 0.01
        time.sleep(WebHandler.base_time)


if __name__ == '__main__':
    tornado.options.parse_command_line()
    application = tornado.web.Application([(r'/', WebHandler)])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
