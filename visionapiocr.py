# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import io
import os
import cv2
import numpy
from copy import deepcopy
import regex
import nltk
import PIL.Image
from wand.image import Image
import argparse
from google.cloud import vision
from google.cloud.vision import types
import pickle
import math
import pytesseract
import string
from nltk.corpus import words as nltk_words
from nltk.tag.stanford import StanfordNERTagger
import csv

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mortgageinfoextraction-2bfca6c82559.json'    # give path to your Service account keys .json
os.environ['STANFORD_CLASSPATH'] = 'stanford/stanford-ner.jar'
os.environ['STANFORD_MODELS'] = 'stanford/classifiers/english.muc.7class.distsim.crf.ser.gz'

class cvrectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

path = ''
dictionary = dict.fromkeys([x.lower() for x in nltk_words.words()], None)


class Utils:
    @staticmethod
    def is_english_word(word):
        # creation of this dictionary would be done outside of
        #     the function because you only need to do it once.
        try:
            x = dictionary[word]
            return True
        except KeyError:
            return False

    @staticmethod
    def is_horizontally_close(rect1, rect2, threshold):
        """
        Check if two rects are horizontally very close to each other
        """
        if (rect1.x + rect1.w)/2 < (rect2.x + rect2.w)/2 and abs(rect2.x - rect1.w) < threshold:
            return True
        elif abs(rect1.x - rect2.w) < threshold:
            return True
        else:
            return False

    @staticmethod
    def is_vertically_close(rect1, rect2, threshold):
        """
        Check if two rects are horizontally very close to each other
        """
        if (rect1.y + rect1.h)/2 < (rect2.y + rect2.h)/2 and abs(rect2.y - rect1.h) < threshold:
            return True
        elif abs(rect1.y - rect2.h) < threshold:
            return True
        else:
            return False

    @staticmethod
    def is_overlap(rect1, rect2):
        if rect1.w < rect2.x or rect1.x > rect2.w or rect1.y > rect2.h or rect1.h < rect2.y:
            return False
        else:
            return True

    @staticmethod
    def is_vertical_overlap(rect1, rect2):
        if rect1.w < rect2.x or rect1.x > rect2.w:
            return False
        else:
            return True

    @staticmethod
    def is_horizontal_overlap(rect1, rect2):
        if rect1.y > rect2.h or rect1.h < rect2.y:
            return False
        else:
            return True

    @staticmethod
    def union(rect1, rect2):
        return cvrectangle(min(rect1.x, rect2.x), min(rect1.y, rect2.y), max(rect1.w, rect2.w), max(rect1.h, rect2.h))

    @staticmethod
    def distance(rect1, rect2):
        x1 = (rect1.x + rect1.w)/2
        y1 = (rect1.y + rect1.h)/2
        x2 = (rect2.x + rect2.w)/2
        y2 = (rect2.y + rect2.h)/2

        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    @staticmethod
    def levenshtein(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = numpy.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix[x, y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1] + 1,
                        matrix[x, y-1] + 1
                    )
        return matrix[size_x - 1, size_y - 1]


class FieldExtraction:
    class Page:
        def __init__(self, page_num=-1, image=None, data=None):
            self.number = page_num
            if data is None:
                self.data = []
            else:
                self.data = data
            self.image = image
            self.raw_text = []
            self.line_data = []
            self.complete_text = {}

        def create_lines(self, multiplier=1, spaces=True):
            self.line_data = []
            for item in self.data:
                r = item['vertices']
                self.line_data.append({'text': item['text'], 'vertices': cvrectangle(r.x, r.y, r.w, r.h)})

            i = 0
            while i < len(self.line_data):
                item1 = self.line_data[i]
                j = i+1
                while j < len(self.line_data):
                    item2 = self.line_data[j]
                    if Utils.is_horizontal_overlap(item1['vertices'], item2['vertices']):
                        if Utils.is_horizontally_close(item1['vertices'], item2['vertices'], multiplier*((item1['vertices'].h - item1['vertices'].y)/2 + (item2['vertices'].h - item2['vertices'].y)/2)):
                            if spaces is True:
                                self.line_data[i]['text'] = self.line_data[i]['text'] + ' ' + self.line_data[j]['text']
                            else:
                                self.line_data[i]['text'] = self.line_data[i]['text'] + self.line_data[j]['text']
                            self.line_data[i]['vertices'] = Utils.union(item1['vertices'], item2['vertices'])
                            self.line_data.remove(self.line_data[j])
                            continue
                    j += 1
                i += 1
            if multiplier < 0.8:
                self.data = self.line_data
                self.line_data = []

        def detect_text(self, pre_load):
            def compare(item1, item2):
                if Utils.is_horizontal_overlap(item1['vertices'],item2['vertices']):
                    return item1['vertices'].x - item2['vertices'].x
                else:
                    return item1['vertices'].y - item2['vertices'].y

            def cmp_to_key(mycmp):
                'Convert a cmp= function into a key= function'
                class K:
                    def __init__(self, obj, *args):
                        self.obj = obj
                    def __lt__(self, other):
                        return mycmp(self.obj, other.obj) < 0
                    def __gt__(self, other):
                        return mycmp(self.obj, other.obj) > 0
                    def __eq__(self, other):
                        return mycmp(self.obj, other.obj) == 0
                    def __le__(self, other):
                        return mycmp(self.obj, other.obj) <= 0
                    def __ge__(self, other):
                        return mycmp(self.obj, other.obj) >= 0
                    def __ne__(self, other):
                        return mycmp(self.obj, other.obj) != 0
                return K
            if pre_load:
                with open(path + ' data ' + str(self.number), 'rb') as input:
                    self.data = pickle.load(input)
                with open(path + ' raw_text ' + str(self.number), 'rb') as input:
                    self.raw_text = pickle.load(input)

                sorted(self.data, key=cmp_to_key(compare))

                self.complete_text = self.data[0]
                self.data.remove(self.complete_text)
                self.raw_text.remove(self.raw_text[0])

                return self.data, self.raw_text

            """Detect text in the file."""
            client = vision.ImageAnnotatorClient()
            # with io.open(self.path, 'rb') as image_file:
            #    content = image_file.read()
            success, encoded_image = cv2.imencode('.png', self.image)
            content = encoded_image.tobytes()

            image = types.Image(content=content)

            response = client.text_detection(image=image)
            texts = response.text_annotations
            self.data = []
            self.raw_text = []

            for text in texts:
                a = ("{}".format(text.description))
                self.raw_text.append(a)
                # print(a)
                vertices = []
                for vertex in text.bounding_poly.vertices:
                    vertices.append((int(vertex.x), int(vertex.y)))
                self.data.append({'text': a, 'vertices': vertices})
            for item in self.data:
                item['vertices'] =cvrectangle(item['vertices'][0][0], item['vertices'][0][1], item['vertices'][2][0], item['vertices'][2][1])

            sorted(self.data, key=cmp_to_key(compare))

            def save_object(obj, filename):
                with open(filename, 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

            save_object(self.data, path + ' data ' + str(self.number))
            save_object(self.raw_text, path + ' raw_text ' + str(self.number))

            self.complete_text = self.data[0]
            self.data.remove(self.complete_text)
            self.raw_text.remove(self.raw_text[0])

            return self.data, self.raw_text

        def detect_text_tesseract(self):
            response = pytesseract.image_to_boxes(self.image)
            for line in response.split('\n'):
                tokens = line.split()
                self.data.append({'text':tokens[0], 'vertices':cvrectangle(int(tokens[1]), self.image.shape[0] - int(tokens[4]), int(tokens[3]), self.image.shape[0] - int(tokens[2]))})
            return self.data

        def visualize_annotations(self):
            x, y = self.image.shape
            temp_data = deepcopy(self.data)
    #        for item in temp_data:
    #            if (((item['vertices'].w - item['vertices'].x) * (item['vertices'].h - item['vertices'].y)) > ((y * x) * .6)):
    #                self.annotated_data.remove(item)
            for item in self.data:
                cv2.rectangle(self.image, (item['vertices'].x, item['vertices'].y), (item['vertices'].w, item['vertices'].h), (100, 255, 34), 4)
            cv2.imwrite(path + "annotated_img.png", self.image)


    def __init__(self, path):
        self.path = path
        self.pages = []
        self.raw_text = []
        self.orgImgs = []

    def addImage(self, img, eroded):
        self.pages.append(self.Page(page_num=len(self.pages), image=eroded))
        self.orgImgs.append(img)

    def create_lines(self):
        for page in self.pages:
            page.create_lines()

    def detect_text(self, pre_load):
        i=0
        for page in self.pages:
            print('Page ', i)
#            if i < 4:
            page.detect_text(pre_load)
#            else:
#                page.detect_text(True)
            #page.visualize_annotations()
            i += 1


    def extract_account_number(self):
        regex_account = "[0-9]{7,11}"
        regex_roll_number = "[A-Z]{0,1}\/[0-9\/]{7,9}.[0-9]{1,2}"
        accounts = []
        rect_acct = cvrectangle(0, 0, 0, 0)
        rect_mortgage = cvrectangle(0, 0, 0, 0)
        isaccountfound = False
        for page in self.pages:
            for item in page.data:
                if (item['text'].upper()).find("ACCOUNT") != -1:
                    rect_acct = item['vertices']
                if (item['text'].upper()).find("MORTGAGE") != -1 and self.pages[0] == page:
                    rect_mortgage = item['vertices']
            """
            txt = regex.search('[0-9]{7,11}', page.complete_text['text'])
            if txt:
                txt = txt.group(0)
                accounts.append(txt)
                isaccountfound = True
                break

            """
            for item in page.data:
                if (regex.search(regex_account, item['text'].strip()) and Utils.is_horizontal_overlap(rect_acct, item['vertices']) and Utils.is_horizontally_close(rect_acct, item['vertices'], 200)):
                    txt = regex.search('[0-9]{7,11}', item['text'])
                    if txt:
                        txt = txt.group(0)

                    accounts.append(txt)
                    isaccountfound = True
                    break
            if isaccountfound is True:
                break

        if len(accounts) ==0:
            for page in self.pages:
                txt = regex.search('[A-Z]{0,1}\/[0-9\/]{7,9}.{1,2}[0-9]{1,2}', page.complete_text['text'])
                if txt:
                    txt = txt.group(0)
                    if txt[-2] != "-":
                            txt = txt[:-2] + '-' + txt[-1:]
                    accounts.append(txt)
                    break

        if len(accounts) == 0:
            accounts.append('NotFound')
        return accounts

    def extract_addresses(self):
        regex_code = regex.compile("^[A-Z150]{1,2}[0-9SIO]([A-Z150]|[0-9SIO])?( [0-9SIO][A-Z150]{3})?$")
        regex_code1 = regex.compile("^[A-Z150]{1,2}[0-9SIO]([A-Z150]|[0-9SIO])?$")
        regex_code2 = regex.compile("^[0-9SIO][A-Z150]{2}$")
        replace_reg = regex.compile('[%s]' % regex.escape(string.punctuation))

        addresses = []

        for page_index in range(0, min(4, len(self.pages))):
            page = self.pages[page_index]
            postal_codes = []
            for item in page.line_data:
                if regex_code.match(replace_reg.sub('', item['text'].upper().strip())):
                    postal_codes.append(item)
            postal_codes1 = []
            postal_codes2 = []
            for item in page.data:
                if regex_code1.match(replace_reg.sub('', item['text'].upper().strip())):
                    postal_codes1.append(item)

                if regex_code2.match(replace_reg.sub('', item['text'].upper().strip())):
                    postal_codes2.append(item)
            for item in postal_codes1:
                for item2 in postal_codes2:
                    if Utils.is_horizontal_overlap(item['vertices'], item2['vertices']):
                        if Utils.is_horizontally_close(item['vertices'],item2['vertices'], 20):
                            code = {'text':'','vertices':cvrectangle(0,0,0,0)}
                            if item['vertices'].w < item2['vertices'].x:
                                code['text'] = item['text'] + ' ' + item2['text']
                            else:
                                continue

                            code['vertices'] = Utils.union(item['vertices'], item2['vertices'])

                            postal_codes.append(code)
            for code in postal_codes:
                for line in page.line_data:
                    if Utils.is_overlap(code['vertices'], line['vertices']):
                        if len(line['text'].split()) < 5:
                            address_segments = [line]
                        else:
                            address_segments = [code]
                        break
                rect = cvrectangle(address_segments[0]['vertices'].x, address_segments[0]['vertices'].y, address_segments[0]['vertices'].w, address_segments[0]['vertices'].h)
                for i in range(0, len(page.line_data)):
                    if not Utils.is_overlap(code['vertices'], page.line_data[i]['vertices']):
                        continue
                    else:
                        for j in range (i-1, -1, -1):
                            rect2 = page.line_data[j]['vertices']
                            if Utils.is_vertical_overlap(rect, rect2):
                                if Utils.is_vertically_close(rect, rect2, abs(rect2.h - rect2.y)*2.5):
                                    if abs(rect.x - rect2.x) < (rect2.h - rect2.y) or abs((rect.x + rect.w)/2 - (rect2.x + rect2.w)/2) < (rect2.h - rect2.y):
                                        address_segments.append(page.line_data[j])
                                        rect = Utils.union(rect, rect2)
                        break
                address = {'text':'' ,'vertices':rect, 'page': page.number}
                for segment in address_segments:
                    if segment != address_segments[-1] or Utils.levenshtein(segment['text'].upper(), 'PROPERTY') >= len('PROPERTY')/3:
                        address['text'] = segment['text'] + ' ' + address['text']
                if len(address['text'].split()) > 3 and len(address['text'].split()) < 12:
                    addresses.append(address)
        return addresses

    def get_property_address(self):
        addresses = self.extract_addresses()

        probs = [0 for x in addresses]      # probability of each address not being the address of the mortgage property

        # TODO : Add flexibility to the comparison criteria of the strings

        i = 0
        minAdd = 0
        for address in addresses:
            closest = 10000000
            for item in self.pages[address['page']].data:
                if Utils.levenshtein(item['text'].upper(), 'PROPERTY') < len('PROPERTY')/3:
                    if closest is None:
                        closest = Utils.distance(item['vertices'], address['vertices'])
                    else:
                        closest = min(closest, Utils.distance(item['vertices'], address['vertices']))
            probs[i] = closest
            if probs[i] < probs[minAdd]:
                minAdd = i
            i += 1

        return addresses[minAdd]

    def get_interest_rate(self):
        selected_items = []
        selected_lines = []
        selected_items_tes = []
        selected_lines_tes = []
        start_page = -1
        start_y = -1
        end_page = -1
        end_y = -1
        test_str1 = '4. DESCRIPTION OF THIS MORTGAGE'
        test_str2 = '5. OVERALL COST OF THIS MORTGAGE'
        for page in self.pages:
            for line in page.line_data:
                if Utils.levenshtein(test_str1, line['text'].upper()) < len(test_str1) / 3 and start_page == -1:
                    start_page = page.number
                    start_y = line['vertices'].h
                if Utils.levenshtein(test_str2, line['text'].upper()) < len(test_str1) / 3:
                    end_page = page.number
                    end_y = line['vertices'].y
                    break

            if end_page != -1:
                break

        # if end_page != -1 and start_page != -1:
        #     tempPages = []
        #     for i in range(0 ,start_page):
        #         tempPages.append(None)
        #     for i in range(start_page, end_page + 1):
        #         page = self.Page(page_num=i, image=self.orgImgs[i])
        #         page.detect_text_tesseract()
        #         page.create_lines()
        #         tempPages.append(page)
        #         cv2.imwrite('asdasdasda.png', self.orgImgs[i])
        #     pages = tempPages
        # else:
        pages = self.pages
        if end_page == -1:
            if start_page != -1:
                end_page = start_page
            else:
                end_page = len(self.pages) - 1
            end_y = 10000
        if start_page == -1:
            start_page = 0

        started = False
        for index in range(start_page, end_page + 1):
            for item in pages[index].data:
                if started is False and item['vertices'].y > start_y:
                    started = True
                if started is True:
                    if (index == end_page and item['vertices'].h > end_y) or index > end_page:
                        break
                    else:
                        x = {'text': item['text'], 'vertices': item['vertices'], 'page': pages[index].number}
                        selected_items.append(x)
            for item in pages[index].line_data:
                if started is False and item['vertices'].y > start_y:
                    started = True
                if started is True:
                    if index == end_page and item['vertices'].h > end_y:
                        break
                    else:
                        x = {'text': item['text'], 'vertices': item['vertices'], 'page': pages[index].number}
                        selected_lines.append(x)
            if start_page - end_page < 3:
                crop = self.orgImgs[index]      #[int(4*item['vertices'].y/3- item['vertices'].h/3):int(5*item['vertices'].h/3 - 2*item['vertices'].y/3), item['vertices'].x - 3:item['vertices'].w + 3]
#                crop = PIL.Image.fromarray(crop.astype('uint8'), mode='L')
#                crop.save('0001.png')
#                x = numpy.array(crop)
#                cv2.imwrite('0000.png', x)
                page = self.Page(page_num=i, image=crop)
                page.detect_text_tesseract()
                page.create_lines(multiplier=0.3, spaces=False)
                page.create_lines()
#                x = {'text': text, 'vertices': item['vertices'], 'page': pages[index].number}
#                selected_items.append(x)
                for item in page.data:
                    if started is False and item['vertices'].y > start_y:
                        started = True
                    if started is True:
                        if (index == end_page and item['vertices'].h > end_y) or index > end_page:
                            break
                        else:
                            x = {'text': item['text'], 'vertices': item['vertices'], 'page': pages[index].number}
                            selected_items_tes.append(x)
                for item in page.line_data:
                    if started is False and item['vertices'].y > start_y:
                        started = True
                    if started is True:
                        if (index == end_page and item['vertices'].h > end_y) or index > end_page:
                            break
                        else:
                            x = {'text': item['text'], 'vertices': item['vertices'], 'page': pages[index].number}
                            selected_lines_tes.append(x)

        rate_regex = regex.compile('^[0-9IlSO]{1,2}([ ]?[.]?[ ]?)[0-9IlSO]{2,3}[ ]?[%]?$')
        candidate_rates = []
        candidate_rates_tes = []
        for item1 in selected_items:
            if rate_regex.match(item1['text']):
                item1['distances'] = 0
                item1['flag'] = True
                candidate_rates.append(item1)
        for item1 in selected_lines:
            if rate_regex.match(item1['text']):
                item1['distances'] = 0
                item1['flag'] = True
                candidate_rates.append(item1)
        for item1 in selected_items_tes:
            if rate_regex.match(item1['text']):
                item1['distances'] = 0
                item1['flag'] = True
                candidate_rates_tes.append(item1)
        for item1 in selected_lines_tes:
            if rate_regex.match(item1['text']):
                item1['distances'] = 0
                item1['flag'] = True
                candidate_rates_tes.append(item1)

        test_str1 = 'INITIAL'
        test_str2 = 'RATE'

        test_str1_items = []
        test_str2_items = []
        for item2 in selected_items:
            if Utils.levenshtein(item2['text'].upper(), test_str1) <= len(test_str1)/3:
                test_str1_items.append(item2)
            elif Utils.levenshtein(item2['text'].upper(), test_str2) <= len(test_str2)/3:
                test_str2_items.append(item2)

        i1 = None
        i2 = None
        mindist = 1000000000
        for item1 in test_str1_items:
            for item2 in test_str2_items:
                if item1['page'] == item2['page']:
                    if Utils.distance(item1['vertices'], item2['vertices']) < mindist:
                        i1 = item1
                        i2 = item2
                        mindist = Utils.distance(item1['vertices'], item2['vertices'])
        if Utils.is_horizontal_overlap(i1['vertices'], i2['vertices']) and Utils.is_horizontally_close(i1['vertices'], i2['vertices'], i1['vertices'].h - i1['vertices'].y):
            pass
        elif Utils.is_vertically_close(i1['vertices'], i2['vertices'], i1['vertices'].h - i1['vertices'].y):
            pass
        else:
            i2 = i1

        for item in candidate_rates:
            if item['page'] == i1['page']:
                if Utils.is_horizontal_overlap(item['vertices'], i1['vertices']):
                    if item['vertices'].x > i1['vertices'].x:
                        item['distances'] += Utils.distance(item['vertices'], i1['vertices'])*0.3
                    else:
                        item['distances'] += Utils.distance(item['vertices'], i1['vertices'])*0.7
                else:
                    rect1 = item['vertices']
                    rect2 = i1['vertices']
                    x1 = (rect1.x + rect1.w)/2
                    y1 = (rect1.y + rect1.h)/2
                    x2 = (rect2.x + rect2.w)/2
                    y2 = (rect2.y + rect2.h)/2
                    item['distances'] += math.sqrt(0.6*((x1-x2)**2) + (y1-y2)**2)
            else:
                item['distances'] += 400
            if item['page'] == i2['page']:
                if Utils.is_horizontal_overlap(item['vertices'], i2['vertices']):
                    if item['vertices'].x > i2['vertices'].x:
                        item['distances'] += Utils.distance(item['vertices'], i2['vertices'])*0.5
                    else:
                        item['distances'] += Utils.distance(item['vertices'], i2['vertices'])*0.7
                else:
                    rect1 = item['vertices']
                    rect2 = i2['vertices']
                    x1 = (rect1.x + rect1.w)/2
                    y1 = (rect1.y + rect1.h)/2
                    x2 = (rect2.x + rect2.w)/2
                    y2 = (rect2.y + rect2.h)/2
                    item['distances'] += math.sqrt(0.6*((x1-x2)**2) + (y1-y2)**2)
            else:
                item['distances'] += 400

        for item in candidate_rates_tes:
            if item['page'] == i1['page']:
                if Utils.is_horizontal_overlap(item['vertices'], i1['vertices']):
                    if item['vertices'].x > i1['vertices'].x:
                        item['distances'] += Utils.distance(item['vertices'], i1['vertices'])*0.3
                    else:
                        item['distances'] += Utils.distance(item['vertices'], i1['vertices'])*0.7
                else:
                    rect1 = item['vertices']
                    rect2 = i1['vertices']
                    x1 = (rect1.x + rect1.w)/2
                    y1 = (rect1.y + rect1.h)/2
                    x2 = (rect2.x + rect2.w)/2
                    y2 = (rect2.y + rect2.h)/2
                    item['distances'] += math.sqrt(0.6*((x1-x2)**2) + (y1-y2)**2)
            else:
                item['distances'] += 400
            if item['page'] == i2['page']:
                if Utils.is_horizontal_overlap(item['vertices'], i2['vertices']):
                    if item['vertices'].x > i2['vertices'].x:
                        item['distances'] += Utils.distance(item['vertices'], i2['vertices'])*0.5
                    else:
                        item['distances'] += Utils.distance(item['vertices'], i2['vertices'])*0.7
                else:
                    rect1 = item['vertices']
                    rect2 = i2['vertices']
                    x1 = (rect1.x + rect1.w)/2
                    y1 = (rect1.y + rect1.h)/2
                    x2 = (rect2.x + rect2.w)/2
                    y2 = (rect2.y + rect2.h)/2
                    item['distances'] += math.sqrt(0.6*((x1-x2)**2) + (y1-y2)**2)
            else:
                item['distances'] += 400

        min = None
        min2 = None
        candidate_rates = sorted(candidate_rates, key=lambda x:x['distances'])
        candidate_rates_tes = sorted(candidate_rates_tes, key=lambda x:x['distances'])
        filter = regex.compile('[1-9]{1}')
        for item in candidate_rates:
            if filter.search(item['text']) and item['text'].find('.') >= 0:
                min = item
                break
        for item in candidate_rates_tes:
            if filter.search(item['text']) and item['text'].find('.') >= 0:
                min2 = item
                break
        if min is None or (min2 is not None and min['distances'] > min2['distances']):
            min = min2

        return min

    def extract_human_names(self, address):
        if address is None:
            return None
        reg = regex.compile('[0-9]')
#        non_english_text = set(self.get_possible_names_stanford(address['page']))
#        non_english_text = list(non_english_text.union(set(self.get_possible_names_nltk(address['page']))))
#        non_english = []
#         for word in self.pages[address['page']].data:
#             # s1 = word['text'].lower()
#             # s1 = re_sub.sub('', s1)
#             # if not Utils.is_english_word(s1):
#             #     if not reg.search(word['text']) and not Utils.is_overlap(address['vertices'], word['vertices']):
#             #         temp = {'text': word['text'], 'vertices': word['vertices'], 'distances': 0}
#             #         non_english.append(temp)
#             for i in range(0, len(non_english_text)):
#                 word2 = non_english_text[i]
#                 if word['text'].find(word2) >= 0:
#                     if Utils.levenshtein(word2.strip().upper(), 'PROPERTY') >= len('PROPERTY')/3:
#                         temp = {'text': word['text'], 'vertices': word['vertices'], 'distances': 0}
#                         non_english.append(temp)
#                     break

        non_english = self.pages[address['page']].data
        ref_str = ['BORROWER', 'CUSTOMER', 'NAME', 'PROPERTY']
        ref_items = [[] for i in ref_str]

        for item in self.pages[address['page']].data:
            i = 0
            for ref in ref_str:
                if Utils.levenshtein(item['text'].upper(), ref) <= len(ref)/3:
                    count = -1
                    for line in self.pages[address['page']].line_data:
                        if Utils.is_overlap(item['vertices'], line['vertices']):
                            count = len(line['text'].split())
                    if (ref == ref_str[3] and count < 8) or count < 4:
                        if ref != 'NAME' or item['text'].upper().find('SAME') < 0:
                            ref_items[i].append(item)
                i += 1
        lens = [len(x) for x in ref_items[:-1]]
        if max(lens) == 0 or len(ref_items[3])==0:
            return None

        minDist = 10000000
        minItem = None
        minItem2 = None
        for i in range(0, len(ref_items) - 1):
            for item in ref_items[i]:
                for item2 in ref_items[len(ref_items)-1]:
                    rect1 = item['vertices']
                    rect2 = item2['vertices']
                    x1 = (rect1.x + rect1.w)/2
                    y1 = (rect1.y + rect1.h)/2
                    x2 = (rect2.x + rect2.w)/2
                    y2 = (rect2.y + rect2.h)/2

                    dist = math.sqrt(0.6*((x1-x2)**2) + (y1-y2)**2)
                    if dist < minDist:
                        minItem = item
                        minItem2 = item2
                        minDist = dist
        ref_item = minItem
        ref_item2 = minItem2

        minItem = None
        minDist = 1000000
        for item in non_english:
            item['distances'] = 0
            if Utils.is_overlap(ref_item['vertices'], item['vertices']) or Utils.is_overlap(ref_item2['vertices'], item['vertices']) or Utils.is_overlap(address['vertices'], item['vertices']):
                continue
            rect1 = item['vertices']
            rect2 = ref_item['vertices']
            x1 = (rect1.x + rect1.w)/2
            y1 = (rect1.y + rect1.h)/2
            x2 = (rect2.x + rect2.w)/2
            y2 = (rect2.y + rect2.h)/2

            if Utils.is_horizontal_overlap(item['vertices'], ref_item['vertices']):
                item['distances'] += math.sqrt(0.1*((x1-x2)**2) + (y1-y2)**2)
            else:
                item['distances'] += math.sqrt(((x1-x2)**2) + (y1-y2)**2)*2
            item['distances'] += Utils.distance(item['vertices'], ref_item2['vertices'])
            if item['distances'] < minDist:
                minItem = item
                minDist = item['distances']

        non_english = sorted(non_english, key=lambda x: x['distances'])

        line = None
        found = False
        for item in non_english:
            for item2 in self.pages[address['page']].line_data:
                if Utils.is_overlap(item['vertices'], item2['vertices']):
                    line = item2
                    if Utils.is_overlap(ref_item['vertices'], line['vertices']) or Utils.is_overlap(ref_item2['vertices'], line['vertices']) or Utils.is_overlap(address['vertices'], line['vertices']) or len(line['text'].split()) > 4:
                        continue
                    else:
                        line['distances'] = item['distances']
                        found = True
                        break
            if found:
                break

        return line

    def detect_sign(self):
        minDist = 1000000
        minItem1 = None
        minItem2 = None
        minLine = None
        minPage = int()
        for page in self.pages:
            ref_str = ['SIGNED', 'BEHALF', 'SIGNATURE']
            ref_items = [[] for i in ref_str]

            for item in page.data:
                i = 0
                for ref in ref_str:
                    if Utils.levenshtein(item['text'].upper(), ref) <= len(ref)/3:
                        count = -1
                        for line in page.line_data:
                            if Utils.is_overlap(item['vertices'], line['vertices']):
                                count = len(line['text'].split())
                        if count > 5 or i==2:
                            ref_items[i].append(item)
                    i += 1

            flag = False
            for item1 in ref_items[0]:
                for item2 in ref_items[1]:
                    rect1 = item1['vertices']
                    rect2 = item2['vertices']
                    if Utils.is_horizontal_overlap(item1['vertices'], item2['vertices']):
                        dist = Utils.distance(rect1, rect2)
                        if dist < minDist:
                            minDist = dist
                            minItem2 = item2
                            minItem1 = item1
                            flag = True

            for item1 in ref_items[2]:
                for item2 in ref_items[1]:
                    rect1 = item1['vertices']
                    rect2 = item2['vertices']
                    dist = Utils.distance(rect1, rect2)
                    if dist < minDist:
                        minDist = dist
                        minItem2 = item2
                        minItem1 = item1
                        flag = True

            if flag:
                for line in page.line_data:
                    if Utils.is_overlap(line['vertices'], minItem1['vertices']) or Utils.is_overlap(line['vertices'], minItem2['vertices']):
                        minLine = line
                        minPage = page.number
                        break

        minPage = self.pages[minPage]

        if minLine is None:
            return False

        prev = minPage.line_data[0]
        prev2 = minPage.line_data[0]
        next = minPage.line_data[-1]
        next2 = minPage.line_data[-1]

        for i in range(0, len(minPage.line_data)):
            if minPage.line_data[i]['vertices'].h < minLine['vertices'].y:
                if minPage.line_data[i]['vertices'].y > prev2['vertices'].y:
                    if minPage.line_data[i]['vertices'].y > prev['vertices'].y:
                        prev2 = prev
                        prev = minPage.line_data[i]
                    else:
                        prev2 = minPage.line_data[i]

            if minPage.line_data[i]['vertices'].y > minLine['vertices'].h:
                if minPage.line_data[i]['vertices'].y < next2['vertices'].y:
                    if minPage.line_data[i]['vertices'].y < next['vertices'].y:
                        next2 = next
                        next = minPage.line_data[i]
                    else:
                        next2 = minPage.line_data[i]

        prev = prev2
        next = next2

        region = cvrectangle(0, 0, page.image.shape[1], 0)
        if abs(prev['vertices'].h - minLine['vertices'].y) > abs(next['vertices'].y - minLine['vertices'].h):
            region.x = min(prev['vertices'].x, minLine['vertices'].x)
            region.w = max(prev['vertices'].w + 50, minLine['vertices'].w + 50, page.image.shape[1])
            region.y = prev['vertices'].h
            region.h = minLine['vertices'].y
        else:
            region.x = min(next['vertices'].x, minLine['vertices'].x)
            region.w = max(next['vertices'].w + 50, minLine['vertices'].w + 50, int(0.8*page.image.shape[1]))
            region.y = minLine['vertices'].h
            region.h = next['vertices'].y

        regionImg = page.image[region.y:region.h, region.x:region.w]
        cv2.imwrite(filepath + 'sign.png', regionImg)
        return True

    def get_possible_names_nltk(self, page):

        person_list = []
        page = self.pages[page]
        tokens = nltk.tokenize.word_tokenize(page.complete_text['text'])
        pos = nltk.pos_tag(tokens)
        sentt = nltk.ne_chunk(pos, binary=False)

        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON' or t.label() == 'ORGANIZATION'):
            for leaf in subtree.leaves():
                person_list.append(leaf[0])
        return person_list

    def get_possible_names_stanford(self, page):

        person_list = []
        page = self.pages[page]
        txt = deepcopy(page.complete_text['text'])
        page.complete_text['text'] = txt.replace("\n", " ")
        tokens = nltk.tokenize.word_tokenize(page.complete_text['text'])
        st = StanfordNERTagger(os.environ['STANFORD_MODELS'], os.environ['STANFORD_CLASSPATH'])
        tagged = st.tag(tokens)
        tagged_sorted = []
        for tag in tagged:
          if tag[1]!='O': tagged_sorted.append(tag)
        person_list = [i[0] for i in tagged_sorted if i[1] == 'ORGANIZATION' or i[1] == 'PERSON']
        #print("person_list", person_list)
        return person_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='image to be processed')
    parser.add_argument('--pdf', help='pdf file to be processed')
    args = parser.parse_args()
    if args.image:
        if (os.path.exists(args.image)):
            print("Processing:", args.image)
            image = cv2.imread(args.image)
            fieldExtraction = FieldExtraction(args.image, image)
            text = fieldExtraction.detect_text()
            fieldExtraction.visualize_annotations()
            fieldExtraction.extract_account_number()
            fieldExtraction.extract_human_names()
    if args.pdf:
        filepath = args.pdf
        # filepath = 'Template_2/template_2_example_24.pdf'
        if (os.path.exists(filepath)):
            print(filepath)
            with Image(filename=filepath, resolution=200) as img:
                page_images = []
                for page_wand_image_seq in img.sequence:
                    page_wand_image = Image(page_wand_image_seq)
                    page_jpeg_bytes = page_wand_image.make_blob(format="jpeg")
                    page_jpeg_data = io.BytesIO(page_jpeg_bytes)
                    page_image = PIL.Image.open(page_jpeg_data)
                    page_images.append(page_image)
        image_list = []
        eroded_list = []
        for page in page_images:
            temp = page.convert('RGB')
            image_list.append(numpy.array(temp))
        file_name = os.path.basename(filepath)

        for i in range(0, len(image_list)):
            image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(str(i) + 'org.png', image_list[i])
            kernel = numpy.ones((4, 2), numpy.uint8)
            dilated = cv2.dilate(image_list[i],kernel,iterations = 1)

            x, dilated = cv2.threshold(dilated, 245, 255, cv2.THRESH_TOZERO)
            kernel = numpy.ones((4, 4), numpy.uint8)
            eroded = dilated
            for j in range(0,5):
                eroded = cv2.erode(eroded,kernel,iterations = 1)
                cv2.max(eroded, image_list[i], eroded)
    #            for x in range(0, eroded.shape[0]):
    #                for y in range(0, eroded.shape[1]):
    #                    eroded[x,y] = max(eroded[x,y], image_list[i][x,y])
            eroded_list.append(eroded)
            cv2.imwrite(str(i) + 'erode.png', eroded_list[i])

        account_list = []
        name_list = []
        path = filepath

        fieldExtraction = FieldExtraction(filepath)

        for i in range(0, len(image_list)):
            fieldExtraction.addImage(image_list[i], eroded_list[i])
        fieldExtraction.detect_text(True)
        fieldExtraction.create_lines()

        fieldExtraction.detect_sign()

        account = fieldExtraction.extract_account_number()
        print("Account", set(account))

        rate = fieldExtraction.get_interest_rate()
        print("Interest Rate:", rate)
        if rate == -1:
            rate = {'text' : "not Found"}

        addresses = fieldExtraction.get_property_address()
        print("Address:", addresses)

        names = fieldExtraction.extract_human_names(addresses)
        if names is None:
            name_list = []
            for page in fieldExtraction.pages:
                box = {'text':'', 'vertices':cvrectangle(0, 0, 0, 0), 'page':page.number}
                temp = fieldExtraction.extract_human_names(box)
                if temp is not None:
                    name_list.append(temp)
            if len(name_list) > 0:
                names = sorted(name_list, key=lambda x: x['distances'])[0]
            else:
                names = {'text': 'Not Found!', 'vertices': cvrectangle(0, 0, 0, 0), 'page': page.number}

        print("Name List", names)

        file_name = file_name + '.csv'
        with open(file_name, 'w') as f:
            writer = csv.writer(f, delimiter ='\t')
            #writer.writerow([names['text'].encode('utf-8'), addresses['text'].encode('utf-8'), account[0], rate['text'].encode('utf-8')])
            #writer.writerow([names['text'].encode('utf-8'), addresses['text'].encode('utf-8'), account[0], rate['text'].encode('utf-8')])
            names['text'] = regex.sub(r'[^\x00-\x7f]',r'', names['text'])
            addresses['text'] = regex.sub(r'[^\x00-\x7f]',r'', addresses['text'])
            rate['text'] = regex.sub(r'[^\x00-\x7f]',r'', rate['text'])
            writer.writerow([names['text'], addresses['text'], account[0], rate['text']])