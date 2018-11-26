#!/usr/bin/python
# coding:utf-8
import os
import time
import codecs
import scrapy
from util import Utilty
from DeepExploit import Msgrpc
from scrapy.http import Request


class SimpleSpider(scrapy.Spider):
    name = 'simple_spider'

    def __init__(self, category=None, *args, **kwargs):
        super(SimpleSpider, self).__init__(*args, **kwargs)
        self.start_urls = getattr(self, 'target_url', None)
        self.allowed_domains = [getattr(self, 'allow_domain', None)]
        self.concurrent = int(getattr(self, 'concurrent', None))
        self.depth_limit = int(getattr(self, 'depth_limit', None))
        self.delay_time = float(getattr(self, 'delay', None))
        self.store_path = getattr(self, 'store_path', None)
        self.response_log = getattr(self, 'response_log', None)
        msgrpc_host = getattr(self, 'msgrpc_host', None)
        msgrpc_port = int(getattr(self, 'msgrpc_port', None))
        self.client = Msgrpc({'host': msgrpc_host, 'port': msgrpc_port})
        self.client.console_id = getattr(self, 'msgrpc_console_id', None).encode('utf-8')
        self.client.token = getattr(self, 'msgrpc_token', None).encode('utf-8')
        self.client.authenticated = True
        self.custom_settings = {
            'CONCURRENT_REQUESTS': self.concurrent,
            'CONCURRENT_REQUESTS_PER_DOMAIN': self.concurrent,
            'DEPTH_LIMIT ': self.depth_limit,
            'DOWNLOAD_DELAY': self.delay_time,
            'ROBOTSTXT_OBEY': True,
            'HTTPCACHE_ENABLED': True,
            'HTTPCACHE_EXPIRATION_SECS': 60 * 60 * 24,
            'HTTPCACHE_DIR': self.store_path,
            'FEED_EXPORT_ENCODING': 'utf-8'
        }
        log_file = os.path.join(self.store_path, self.response_log)
        self.fout = codecs.open(log_file, 'w', encoding='utf-8')
        Utilty().print_message('ok', 'Save log to {}'.format(log_file))

    def start_requests(self):
        self.client.keep_alive()
        url = self.start_urls
        yield Request(url, self.parse)

    def parse(self, response):
        self.fout.write(response.body.decode('utf-8'))
        for href in response.css('a::attr(href)'):
            full_url = response.urljoin(href.extract())
            time.sleep(self.delay_time)
            yield scrapy.Request(full_url, callback=self.parse_item)
        for src in response.css('script::attr(src)'):
            full_url = response.urljoin(src.extract())
            time.sleep(self.delay_time)
            yield scrapy.Request(full_url, callback=self.parse_item)

    def parse_item(self, response):
        self.client.keep_alive()
        urls = []
        self.fout.write(response.body.decode('utf-8'))
        for href in response.css('a::attr(href)'):
            full_url = response.urljoin(href.extract())
            urls.append(full_url)
        for src in response.css('script::attr(src)'):
            full_url = response.urljoin(src.extract())
            urls.append(full_url)
        yield {
            'urls': urls,
        }
