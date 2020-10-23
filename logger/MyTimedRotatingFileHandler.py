try:
    import codecs
except ImportError:
    codecs = None
import logging.handlers
import time
import os

class MyTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
  def __init__(self,dir_log):
   self.dir_log = dir_log
   filename =  self.dir_log+time.strftime("%m%d%Y") #dir_log here MUST be with os.sep on the end
   logging.handlers.TimedRotatingFileHandler.__init__(self,filename, when='midnight', interval=1, backupCount=0, encoding=None)
  def doRollover(self):
   """
   TimedRotatingFileHandler remix - rotates logs on daily basis, and filename of current logfile is time.strftime("%m%d%Y")+".txt" always
   """
   self.stream.close()
   # get the time that this sequence started at and make it a TimeTuple
   t = self.rolloverAt - self.interval
   timeTuple = time.localtime(t)
   self.baseFilename = self.dir_log+time.strftime("%m%d%Y")
   if self.encoding:
     self.stream = codecs.open(self.baseFilename, 'w', self.encoding)
   else:
     self.stream = open(self.baseFilename, 'w')
   self.rolloverAt = self.rolloverAt + self.interval