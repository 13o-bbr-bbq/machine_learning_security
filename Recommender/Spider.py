#!/usr/bin/python
# coding:utf-8
import time
import scrapy
from scrapy.http import Request


class SimpleSpider(scrapy.Spider):
    name = 'simple_spider'

    def __init__(self, category=None, *args, **kwargs):
        super(SimpleSpider, self).__init__(*args, **kwargs)
        self.start_urls = getattr(self, 'target_url', None)
        self.allowed_domains = [getattr(self, 'allow_domain', None)]
        self.delay_time = float(getattr(self, 'delay', None))
        self.custom_settings = {
            'DEPTH_LIMIT ': 5,
            'DOWNLOAD_DELAY': self.delay_time,
            'ROBOTSTXT_OBEY': True,
            'FEED_EXPORT_ENCODING': 'utf-8'
        }

    def start_requests(self):
        url = self.start_urls
        yield Request(url, self.parse)

    def parse(self, response):
        for href in response.css('a::attr(href)'):
            full_url = response.urljoin(href.extract())
            time.sleep(self.delay_time)
            yield scrapy.Request(full_url, callback=self.parse_item)

    def parse_item(self, response):
        urls = []
        for href in response.css('a::attr(href)'):
            full_url = response.urljoin(href.extract())
            urls.append(full_url)
        yield {
            'urls': urls,
        }
