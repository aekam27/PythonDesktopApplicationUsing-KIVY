#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 00:09:03 2020

@author: adsorbentkarma
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")  
#model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
import nltk 
import random
nltk.download('wordnet')
nltk.download('stopwords')
#from nltk.corpus import wordnet 
#from nltk.corpus import stopwords
#from nltk.tokenize import sent_tokenize, word_tokenize 
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile 
from num2words import num2words 
#golvefile = datapath('/Users/adsorbentkarma/Desktop/glove.6B/glove.6B.100d.txt')
#w2v = get_tmpfile("glove.6B.100d.word2vec.txt")
#glove2word2vec(golvefile,w2v)
#model = KeyedVectors.load_word2vec_format(w2v)
#import spacy#
#from spellchecker import SpellChecker
#spell = SpellChecker(distance=2) 

#nlp = spacy.load("en_core_web_lg")
#SymD = {"~":"Tilde","!":"bang","@":"at the rate","#":"hash","$":"dollar","%":"percent","+":"plus"}
#alreadydone = {}
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.progressbar import ProgressBar 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import cv2
from imutils import paths
import random
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
#from itertools import permutations 
#from keras.models import load_model
#from tensorflow.keras.preprocessing.image import img_to_array
#from gensim.models import Word2Vec
import os
import speech_recognition as sr 
#import numpy as np
#import keras.backend.tensorflow_backend as tb
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config 
#import speech_recognition as sr
#import multiprocessing
import time
summarizer = pipeline("summarization")
Config.set('graphics', 'resizable', '1') 
   
Config.set('graphics', 'width', '800') 
   
Config.set('graphics', 'height', '500') 

from kivy.metrics import dp
from kivymd.app import MDApp
#import kivymd.uix.datatables
from kivymd.uix.datatables import MDDataTable
from kivy.uix.anchorlayout import AnchorLayout
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from pymongo import MongoClient
import datetime 
import pandas as pd
from kivymd.uix.button import MDRectangleFlatButton
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
#from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.popup import Popup
#from kivy.uix.image import Image
# The Label widget is for rendering text.  
from kivy.uix.gridlayout import GridLayout   
from kivy.uix.textinput import TextInput
from rivescript import RiveScript
from playsound import playsound
import multiprocessing 
from paraphrase_googletranslate import Paraphraser

r = sr.Recognizer()
client = MongoClient("#")
db = client["#"]
rs = RiveScript()
rs.load_directory("Brain")
rs.sort_replies()   
print("ok")
Path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 
from gtts import gTTS


class SalesTable(MDApp):
    
    def build(self):
        #screen = Screen()
        #layout = AnchorLayout()
        data = db.AggrW.find()
        reda = []
        for x in data:
            ti=[]
            for i,j in x.items():
                if i == "ProductCategory":
                    ti.append(j)
                elif i=="ProductName":
                    ti.append(j)
                elif i =="ProductWidth":
                    ti.append(j)
                elif i=="ProductLength":
                    ti.append(j)
                elif i=="ProductSoldTo":
                    ti.append(j)
                elif i=="ProductSoldOn":
                    ti.append(j)
                elif i=="ProductQuantitySold":
                    ti.append(j)
            reda.append(tuple(ti))
        #print(reda)
        self.data_tables = MDDataTable(
            size_hint=(0.9, 0.6),
            use_pagination= False,
            check=True,
            rows_num = 100000,
            column_data=[
                ("ProductCategory", dp(45)),
                ("ProductName", dp(45)),
                ("ProductWidth", dp(45)),
                ("ProductLength", dp(45)),
                ("ProductSoldTo", dp(45)),
                ("ProductSoldOn", dp(45)),
                ("ProductQuantitySold", dp(45)),
            ],
            row_data=reda,
        )
        self.data_tables.bind(on_row_press=self.on_row_press)
        self.data_tables.bind(on_check_press=self.on_check_press)
        #layout.add_widget(self.data_tables)
        #screen.add_widget(layout)
        #screen.add_widget(
        #    MDRectangleFlatButton(
        #        text="Hello, World",
        #        pos_hint={"center_x": 0.9, "center_y": 0.9},
        #        on_press=self.on_stop
        #    )
        #)
        #return screen
    def on_start(self):
        self.data_tables.open()

    def on_row_press(self, instance_table, instance_row):
        '''Called when a table row is clicked.'''

    def on_check_press(self, instance_table, current_row):
        '''Called when the check box in the table row is checked.'''
        #print(current_row[0],current_row[1],current_row[2],current_row[3],current_row[4],current_row[5],current_row[6])
        db.AggrW.delete_one({"ProductCategory":current_row[0],"ProductName":current_row[1],"ProductWidth":current_row[2],"ProductLength":current_row[3],"ProductSoldTo":current_row[4],"ProductQuantitySold":current_row[6]})
        #print("DONE")
    #def on_stop(self):
        #SalesTable.stop()
        


class StockTable(MDApp):
    def build(self):
        data = db.AggrWS.find()
        reda = []
        for x in data:
            ti=[]
            for i,j in x.items():
                if i == "ProductCategory":
                    ti.append(j)
                elif i=="ProductName":
                    ti.append(j)
                elif i =="ProductWidth":
                    ti.append(j)
                elif i=="ProductLength":
                    ti.append(j)
                elif i=="ProductQuantityAvailable":
                    ti.append(j)
            reda.append(ti)
        self.data_tables = MDDataTable(
            size_hint=(0.9, 0.6),
            use_pagination= False,
            check=False,
            rows_num = 100000,
            column_data=[
                ("ProductCategory", dp(45)),
                ("ProductName", dp(45)),
                ("ProductWidth", dp(45)),
                ("ProductLength", dp(45)),
                ("ProductQuantityAvailable", dp(45)),
            ],
            row_data=reda,
        )
        self.data_tables.bind(on_row_press=self.on_row_press)
        self.data_tables.bind(on_check_press=self.on_check_press)

    def on_start(self):
        self.data_tables.open()

    def on_row_press(self, instance_table, instance_row):
        '''Called when a table row is clicked.'''

        print(instance_table, instance_row)

    def on_check_press(self, instance_table, current_row):
        '''Called when the check box in the table row is checked.'''
        #print(current_row[0],current_row[1],current_row[2],current_row[3],current_row[4],current_row[5],current_row[6])
        db.AggrW.delete_one({"ProductCategory":current_row[0],"ProductName":current_row[1],"ProductWidth":current_row[2],"ProductLength":current_row[3],"ProductSoldTo":current_row[4],"ProductQuantitySold":current_row[6]})
        #print("DONE")
        #print(instance_table, current_row)

class StartScreen(Screen):
    pass

class Exit(Screen):
    pass

class About(Screen):
    pass

class ARTG(Screen):
    def Art(self):
        set_seed(42)
        i = self.ids.inp.text
        o = generator(i, max_length=350, num_return_sequences=1)[0]["generated_text"]
        self.ids.out.text = o

class SUMMA(Screen):
    
    def summa(self):
        i = self.ids.inppp.text
        o = summarizer(i, max_length=200, min_length=30)
        self.ids.outtt.text = o[0]['summary_text']
        


class PARA(Screen):
  
    def para(self):
        phraser = Paraphraser(random_ua=True)
        rephrased = phraser.paraphrase(self.ids.inpp.text)
        #sentence = self.ids.inpp.text
        #text =  "paraphrase: " + sentence + " </s>"
        #encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
        #input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        #outputs = model.generate(
        #        input_ids=input_ids, attention_mask=attention_masks,
        #        max_length=256,
        #        do_sample=True,
        #        top_k=120,
        #        top_p=0.95,
        #        early_stopping=True,
        #        num_return_sequences=1
        #        )
        #for output in outputs:
        #    line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            #print(line)
        #    break
        #self.ids.outt.text = line

        self.ids.outt.text = str(rephrased)

class Register(Screen):

    def show_load(self):
        self.layout = BoxLayout()
        self.lay = BoxLayout()
        Bu = Button(text="Load I")
        BUV = Button(text="Load V")
        FSC = FileChooserIconView(dirselect= True)
        self.lay.add_widget(Bu)
        self.lay.add_widget(BUV)        
        self.layout.add_widget(FSC)
        self.layout.add_widget(self.lay)
        Bu.bind(on_press =  lambda x:(self.load(FSC.path, FSC.selection)))
        BUV.bind(on_press =  lambda x:(self.loadVideo(FSC.path, FSC.selection)))
   
        self.popup = Popup(title="Load file", content=self.layout,
                            size_hint=(0.8, 0.8))
        self.popup.open()  
        
    
    def record(self):

        self.newPathDESKV = os.path.join(Path,"TEMP")
        if not os.path.exists(self.newPathDESKV):
            os.mkdir(self.newPathDESKV)
        capture_duration = 2
        vid = cv2.VideoCapture(0) 
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        n = random.randint(0,500)
        
        out = cv2.VideoWriter(self.newPathDESKV+"/"+str(n)+'.avi', fourcc, 30.0, (800, 400))
        start_time = time.time()
        while( int(time.time() - start_time) < capture_duration ):
            ret, frame = vid.read() 
            if ret == True:  
                frame = cv2.resize(frame, (800, 400))
                out.write(frame)
                cv2.imshow('frame', frame)  
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
        vid.release() 
        out.release()
        cv2.destroyAllWindows()  
        fname = str(n)+'.avi'
        self.loadVideo(self.newPathDESKV, fname)
        
        
    def filestart(self,tiu):
        try:
            knownEncodings = [] 
            knownNames = []   
            #$for (i, imagePath) in enumerate(imagePaths):
            #print(self.imagePaths[0])
            # print("[INFO] processing image {}/{}".format(i + 1,len(self.imagePaths)))
            name = self.imagePaths[0].split(os.path.sep)[-2]
            #print(name)
            image = cv2.imread(self.imagePaths[0])
            #print(image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample= 2 )
            encods = face_recognition.face_encodings(rgb, boxes, num_jitters=100)
            for enco in encods:
                knownEncodings.append(enco)
                knownNames.append(name)
            data = pickle.loads(open('aekam.pickle', "rb").read())
            key_list = list(data.keys())
            #print(key_list,data[key_list[0]],data[key_list[1]])
            data[key_list[0]] = data[key_list[0]]+knownEncodings
            data[key_list[1]] = data[key_list[1]]+knownNames
            f = open("aekam.pickle", "wb")
            f.write(pickle.dumps(data))
            f.close()
            
        except:
            Clock.unschedule(self.t1)
            Clock.unschedule(self.t2)
            Clock.unschedule(self.t3)
            self.popupp.dismiss()
            try:
                os.rmdir(self.newP)
                
            except:
                pass
            self.ids.fff.text = "DONE"


    def load(self, path, filename):
        try:
            self.popup.dismiss()
        except:
            pass
        self.progress_bar = ProgressBar()
        self.popupp = Popup(
            title ='Encoding Images',
            content = self.progress_bar
        )
        self.progress_bar.value = 1
        #self.popup.bind(on_open = Clock.schedule_interval(self.prob, 1 / 25))
        self.popupp.open()
        try:
            self.pathtoimages = os.path.join(path, filename)
            
        except:
            self.pathtoimages = os.path.join(path, filename[0])
            
        self.imagePaths = list(paths.list_images(self.pathtoimages))
        #print(self.pathtoimages)
        self.len = list(paths.list_images(self.pathtoimages))
        self.k = len(self.len)
        #self.filestart()
        #rint(self.imagePaths)
        self.t1 = Clock.schedule_interval(self.filestart ,2)
        self.t2 = Clock.schedule_interval(self.popList,2)
        self.t3 = Clock.schedule_interval(self.prob,2)

            
    def loadVideo(self, path, filename):
        try:
            self.popup.dismiss()
        except:
            pass
        
        self.pathtovideofol = path
        try:
            self.pathtovideo = os.path.join(path, filename)
        except:
            self.pathtovideo = os.path.join(path, filename[0])
            
        layout = FloatLayout()
        Lab = Label(text = "Please Enter Your Name",pos_hint={'x':0,'y':0},size_hint=(1, .7),font_size='20sp')
        layout.add_widget(Lab)
        def newdir(tey):
            message = str(tey.text)
            self.videotoimage(message)
            

        
        SubmitButton = Button(text = "GO",pos_hint={'x':.7,'y':0},size_hint=(.3, .2)) 
        textinput = TextInput(pos_hint={'x':0,'y':0},size_hint=(.7, .2),font_size='20sp')
        layout.add_widget(textinput)
        layout.add_widget(SubmitButton)
        
        SubmitButton.bind(on_press = lambda x:newdir(textinput))
        self.popupv = Popup(title="Enter Your Name", content=layout,
                            size_hint=(0.69, 0.69))
        self.popupv.open()  

    def videotoimage(self,name):
        self.popupv.dismiss()
        self.newP = os.path.join(self.pathtovideofol,name)
        os.mkdir(self.newP)
        cap= cv2.VideoCapture(self.pathtovideo)
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(os.path.join(self.newP, name+str(i)+'.jpg'),frame)
            i+=1
        cap.release()
        cv2.destroyAllWindows() 
        self.load(self.pathtovideofol,name)
        
        
    def popList(self,yu):
        try:
            self.imagePaths.pop(0)
        except:
            Clock.unschedule(self.t2)
        
    def prob(self,ti):
        if self.progress_bar.value>= 100:
            return False
        self.progress_bar.value += int(100/self.k)

    
class Cameras(Screen):

    def build(self):
        k = self.ids.url.text
        k2 = self.ids.url2.text
        if k.isdigit():
            k = int(k)
        if k2.isdigit():
            k2 = int(k2)
        self.vs = VideoStream(src=k).start()
        self.vs1 = VideoStream(src=k2).start()
        self.c1 = Clock.schedule_interval(self.update, 1.0/9000000000.0)
        self.c2 = Clock.schedule_interval(self.update1, 1.0/900000000.0)
   

    def update(self,dtt):
        try:
            self.ids.ff.text = ""
            data = pickle.loads(open('MARCH14HOG.pickle', "rb").read())
            frame = self.vs.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width = 400)
            r = frame.shape[1] / float(rgb.shape[1])          
            boxes = face_recognition.face_locations(rgb, model= 'hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],encoding, tolerance = 0.43)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}      
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 100), 5)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 70), 8)
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.img1.texture = texture1
        except:
            Clock.unschedule(self.c1)
            texx = Texture.create(size=(600, 600), colorfmt='bgr')
            self.ids.img1.texture= texx
            self.ids.ff.text = "NEECHE BOX MAI URL LINK DAAL"
        
    def update1(self,dtt):
        try:
            self.ids.ff.text = ""
            data = pickle.loads(open('MARCH14HOG.pickle', "rb").read())
            frame = self.vs1.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width = 400)
            r = frame.shape[1] / float(rgb.shape[1])
            boxes = face_recognition.face_locations(rgb, model= 'hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],encoding, tolerance = 0.43)
                name = "Unknown"
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}      
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                cv2.rectangle(frame, (left, top), (right, bottom),(112, 255, 11), 5)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (12, 255, 0), 8)
            buf2 = cv2.flip(frame, 0)
            buf2 = buf2.tostring()
            texture2 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            texture2.blit_buffer(buf2, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.img2.texture = texture2
        
        except:
            Clock.unschedule(self.c2)
            texxx = Texture.create(size=(600, 600), colorfmt='bgr')
            self.ids.img2.texture= texxx
            self.ids.ff.text = "NEECHE BOX MAI URL LINK DAAL"
    
    def stopppp(self):
        self.ids.url.text = ""
        self.ids.url2.text =""
        self.build()
        

class MainWindow(Screen):
    def StockDown(self):
        data = db.AggrWS.find()
        reda = []
        for x in data:
            ti=[]
            for i,j in x.items():
                if i == "ProductCategory":
                    ti.append(j)
                elif i=="ProductName":
                    ti.append(j)
                elif i =="ProductWidth":
                    ti.append(j)
                elif i=="ProductLength":
                    ti.append(j)
                elif i=="ProductQuantityAvailable":
                    ti.append(j)
            reda.append(ti)
        DF = pd.DataFrame(reda)
        DF.to_csv(os.path.join(Path,'StockExcel.csv'), index=False, header=["ProductCategory",'ProductName', 'ProductWidth','ProductLength','ProductQuantityAvailable'])
        os.system('say Download Completed')
    def SaleDown(self):
          data = db.AggrW.find()
          reda = []
          for x in data:
              ti=[]
              for i,j in x.items():
                  if i == "ProductCategory":
                      ti.append(j)
                  elif i=="ProductName":
                      ti.append(j)
                  elif i =="ProductWidth":
                      ti.append(j)
                  elif i=="ProductLength":
                      ti.append(j)
                  elif i=="ProductSoldTo":
                      ti.append(j)
                  elif i=="ProductSoldOn":
                      ti.append(j)
                  elif i=="ProductQuantitySold":
                      ti.append(j)
              reda.append(ti)
          DF = pd.DataFrame(reda)
          DF.to_csv(os.path.join(Path,'SalesExcel.csv'), index=False, header=["ProductCategory",'ProductName', 'ProductWidth','ProductLength', 'ProductSoldTo', 'ProductSoldOn', 'ProductQuantitySold'])
          os.system('say Download Completed')
    def mainman(self,text):
        if text=='Stock Excel':
            self.StockDown()
        elif text=='Sales Excel':
            self.SaleDown() 
    
    def botpopup(self):
        
        layout = FloatLayout()
        Lab = Label(text = " ",pos_hint={'x':0,'y':0},size_hint=(1, .7),font_size='20sp')
        layout.add_widget(Lab)

        def aml(textt):
            try:
                message = str(textt.text)
            except:
                message=textt
            #print(message)
            while True:
                response = rs.reply("localuser", message)
                print(response)
                if response =="[ERR: No Reply Matched]":
                    try:
                        #Lab = Label(text = "HAHALOLTRY")
                        tts = gTTS("Chl Chl Baap Ko Mut Sikha", lang='hi')
                        tts.save("DEResponse.mp3")
                        
                        Lab.text=str("Chl Chl Baap Ko Mat Sikha")
                        try:
                            playsound("DEResponse.mp3") 
                        except:
                            pass
                        break
                    except:
                        #Lab = Label(text = "HAHALOLExcept")
                        #layout.add_widget(Lab)
                        break
                else:
                    tts = gTTS(text=response, lang='en')
                    tts.save("Response.mp3")
                    Lab.text=str(response)
                    try:
                        playsound("Response.mp3") 
                    except:
                        pass
                    #Lab = Label(text = response)
                    #layout.add_widget(Lab)
                    break
                
                    #return response
                    
        def lstn2():
            try: 
                with sr.Microphone() as source1: 
                    r.adjust_for_ambient_noise(source1, duration=0.1)   
                    audio1 = r.listen(source1) 
                    t = r.recognize_google(audio1) 
                    t = t.lower()
                    aml(t)
            except: 
                tts = gTTS("Theek Se Bol Samajh Nhi Aaya", lang='hi')
                tts.save("DEEEResponse.mp3")
                 
                Lab.text=str("Theek Se Bol Samajh Nhi Aaya")
                try:
                    playsound("DEEEResponse.mp3")
                except:
                    pass
            
        SubmitButton = Button(text = "GO",pos_hint={'x':.7,'y':0},size_hint=(.2, .2)) 
        textinput = TextInput(pos_hint={'x':0,'y':0},size_hint=(.7, .2),font_size='20sp')
        layout.add_widget(textinput)
        layout.add_widget(SubmitButton)
        popup = Popup(title ='LANA', content = layout, size_hint= (.69,.69)) 
        popup.open()
        VoiceButton = Button(text='VOICE',pos_hint={'x':.9,'y':0},size_hint=(.1, .2))
        layout.add_widget(VoiceButton)
        
        SubmitButton.bind(on_press = lambda x:aml(textinput))
        VoiceButton.bind(on_press = lambda x:lstn2())
        
        
class WindowManager(ScreenManager):
    def changescreen(self, value):

        try:
            if value=='EXIT':
                JhonnySin().stop()
            elif value not in ['|||','Update']:
                self.current = value
        except:
            print('No screen named ' + value)

    
class SalesScreen(Screen):
    pc= ObjectProperty(None)
    pn = ObjectProperty(None)
    pw = ObjectProperty(None)
    pl = ObjectProperty(None)
    pst = ObjectProperty(None)
    pqs = ObjectProperty(None)
    def Sales(self):
        pc =  self.pc.text
        pn =  self.pn.text
        pw =  self.pw.text
        pl =  self.pl.text
        pst =  self.pst.text
        pso = datetime.datetime.now()
        pqs =  self.pqs.text
        db.AggrW.insert_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl,'ProductSoldTo':pst, 'ProductSoldOn':pso,'ProductQuantitySold':pqs})
        try:
            for x in db.AggrWS.find_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl}):
                t = x['ProductQuantityAvailable']
                nq = int(t)-int(pqs)
                db.AggrWS.insert_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl,'ProductQuantityAvailable':nq})
                os.system('say Updated')
        except:
            print("ok2")
            newq = 0-int(pqs)
            db.AggrWS.insert_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl,'ProductQuantityAvailable':newq})
            os.system('say Updated')

class StockScreen(Screen):
    pc= ObjectProperty(None)
    pn = ObjectProperty(None)
    pw = ObjectProperty(None)
    pl = ObjectProperty(None)
    pq = ObjectProperty(None)
    def Stocks(self):
        pc =  self.pc.text
        pn =  self.pn.text
        pw =  self.pw.text
        pl =  self.pl.text
        pq =  self.pq.text
        try:
             for x in db.AggrWS.find_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl}):
                 print("Ok4")
                 t = x['ProductQuantityAvailable']
                 print(t)
                 nq = int(t)+int(pq)
                 print("Ok5")
                 db.AggrWS.insert_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl,'ProductQuantityAvailable':nq})
                 print("Ok6")
                 os.system('say Updated')

        except:
            print("Ok7")
            newq = int(pq)
            print("Ok8")
            db.AggrWS.insert_one({"ProductCategory":pc ,'ProductName':pn, 'ProductWidth':pw,'ProductLength':pl,'ProductQuantityAvailable':newq})
            print("Ok9")
            os.system('say Updated')


class ViewTables(Screen):
    #def build(self):
    #    self.vs = VideoStream(src=0).start()
    #    Clock.schedule_once(self.update, 1.0/33.0)      
    #def update(self,dt):
    #    data = pickle.loads(open('MARCH14HOG.pickle', "rb").read())
    #    while True: 
    #        frame = self.vs.read()
    #        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        rgb = imutils.resize(frame, width = 400)
    #        r = frame.shape[1] / float(rgb.shape[1])          
    #        boxes = face_recognition.face_locations(rgb, model= 'hog')
    #        encodings = face_recognition.face_encodings(rgb, boxes)
    #        names = []
    #        for encoding in encodings:
    #            matches = face_recognition.compare_faces(data["encodings"],encoding, tolerance = 0.43)
    #            name = "Unknown"
    #            if True in matches:
    #                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    #                counts = {}      
    #                for i in matchedIdxs:
    #                    name = data["names"][i]
    #                    counts[name] = counts.get(name, 0) + 1
    #                name = max(counts, key=counts.get)
    #            names.append(name)
    #        for ((top, right, bottom, left), name) in zip(boxes, names):
    #            top = int(top * r)
    #            right = int(right * r)
    #            bottom = int(bottom * r)
    #            left = int(left * r)
    #            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
    #            y = top - 15 if top - 15 > 15 else top + 15
    #            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    #        cv2.imshow("Frame", frame)
    #        key = cv2.waitKey(1) & 0xFF
    #        if key == ord("q"):
    #            break        
    #        if "Aekam" in names:
    #            cv2.destroyAllWindows() 
    #            self.vs.stop()
    #            StockTable().run()
    #            self.vs.stream.release()
    #            break
    #def buildd(self):
    #    SalesTable().run()
        #self.vs = VideoStream(src=0).start()
        #Clock.schedule_once(self.updatee, 1.0/33.0)   
    #def updatee(self,dt):
    #    data = pickle.loads(open('MARCH14HOG.pickle', "rb").read())
    #    while True: 
    #        frame = self.vs.read()
    #        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        rgb = imutils.resize(frame, width = 400)
    #        r = frame.shape[1] / float(rgb.shape[1])          
    #        boxes = face_recognition.face_locations(rgb, model= 'hog')
    #        encodings = face_recognition.face_encodings(rgb, boxes)
    #        names = []
    #        for encoding in encodings:
    #            matches = face_recognition.compare_faces(data["encodings"],encoding, tolerance = 0.43)
    #            name = "Unknown"
    #            if True in matches:
    #                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    #                counts = {}      
    #                for i in matchedIdxs:
    #                    name = data["names"][i]
    #                    counts[name] = counts.get(name, 0) + 1
    #                name = max(counts, key=counts.get)
    #            names.append(name)
    #        for ((top, right, bottom, left), name) in zip(boxes, names):
    #            top = int(top * r)
    #           right = int(right * r)
    #            bottom = int(bottom * r)
    #            left = int(left * r)
    #            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
    #            y = top - 15 if top - 15 > 15 else top + 15
    #            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    #    
    #        cv2.imshow("Frame", frame)
    #        key = cv2.waitKey(1) & 0xFF
    #        if key == ord("q"):
    #            break        
    #        if "Aekam" in names:
    #            cv2.destroyAllWindows() 
    #            self.vs.stop()
    #            SalesTable().run()
    #            self.vs.stream.release()
    #            break
   # def StockTable(self):            
        #print(MDApp.get_running_app())
   #     StockTable().stop()
   #     JhonnySin().stop()
        #print(MDApp.get_running_app())
   #     try:
   #         SalesTable().run()
   #         StockTable().run()
        #print(MDApp.get_running_app())
   #     except:
   #         SalesTable().run()
   #         StockTable().run()
            
    #def SaleTable(self):
        #print(MDApp.get_running_app())
   #     SalesTable().stop()
   #     JhonnySin().stop()
   #     try:
   #         SalesTable().stop()
   #         SalesTable().run()
       #print(MDApp.get_running_app())
   #     except:
   #         SalesTable().stop()
   #         SalesTable().run()
            
    def NUMPAD(self,instance):
        layout = GridLayout(cols=2)
        popup = Popup(title ='Enter Code', content = layout, size_hint= (.69,.69),title_size='30sp')
        def passw(bal,txt):
            a = str(bal.text)
            b = textinput.text
            print(b)
            textinput.text = b+a
            if textinput.text == "0000":
                if txt == 'Sales Table':
                    popup.dismiss()
                    SalesTable().run()
                else:
                    popup.dismiss()
                    StockTable().run()
            elif len(textinput.text)>4:
                popup.dismiss()

        textinput = TextInput(password=True,font_size='50sp',readonly= True)
        textinput1 = TextInput(password=True,font_size='50sp',readonly= True)
        layout.add_widget(textinput)
        layout.add_widget(textinput1)
        popup.open()
        num = ['num0','num1','num2','num3','num4','num5','num6','num7','num8','num9']
        for i in range(0,10):
            num[i] = Button(text=str(i))
            layout.add_widget(num[i])
        num[0].bind(on_press = lambda x:passw(num[0],instance.text))
        num[1].bind(on_press = lambda x:passw(num[1],instance.text))
        num[2].bind(on_press = lambda x:passw(num[2],instance.text))
        num[3].bind(on_press = lambda x:passw(num[3],instance.text))
        num[4].bind(on_press = lambda x:passw(num[4],instance.text))
        num[5].bind(on_press = lambda x:passw(num[5],instance.text))
        num[6].bind(on_press = lambda x:passw(num[6],instance.text))
        num[7].bind(on_press = lambda x:passw(num[7],instance.text))
        num[8].bind(on_press = lambda x:passw(num[8],instance.text))
        num[9].bind(on_press = lambda x:passw(num[9],instance.text))
        
    
class abcd(App):
    def build(self):
        return kv



if __name__ == '__main__':
    print("QWO")
    kv = Builder.load_file("abcd.kv") 
    abcd().run()
    
