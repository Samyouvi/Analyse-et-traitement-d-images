# -*- coding: utf-8 -*-

from __future__ import print_function

from scipy import spatial
from sklearn.utils.extmath import weighted_mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import collections

class Evaluation(object):

  def make_samples(self):
    raise NotImplementedError("Needs to implemented this method")


def distance(v1, v2, d_type='d1'):
  assert v1.shape == v2.shape, "shape of two vectors need to be same!"

  if d_type == 'd1':
    return np.sum(np.absolute(v1 - v2))
  elif d_type == 'd2':
    return np.sum((v1 - v2) ** 2)
  elif d_type == 'd2-norm':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd3':
    pass
  elif d_type == 'd4':
    pass
  elif d_type == 'd5':
    pass
  elif d_type == 'd6':
    pass
  elif d_type == 'd7':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd8':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'cosine':
    return spatial.distance.cosine(v1, v2)
  elif d_type == 'square':
    return np.sum((v1 - v2) ** 2)


def AP(label, results, sort=True):
  ''' infer a query, return it's ap

    arguments
      label  : query's class
      results: a dict with two keys, see the example below
               {
                 'dis': <distance between sample & query>,
                 'cls': <sample's class>
               }
      sort   : sort the results by distance
  '''
  if sort:
    results = sorted(results, key=lambda x: x['dis'])
  precision = []
  hit = 0
  for i, result in enumerate(results):
    if result['cls'] == label:
      hit += 1
      precision.append(hit / (i+1.))
  if hit == 0:
    return 0.
  return np.mean(precision)

def myAP(label, results, sort=True):
    if sort:
        results = sorted(results, key=lambda x: x['dis'])
    precision = []

    for i, result in enumerate(results):
        hit = 0.
        if result['cls'] == label:
            hit = 1.
            precision.append(hit)
            
    return (np.mean(precision)>0.5)

def infer(query, samples=None, db=None, sample_db_fn=None, depth=None, d_type='d1'):
  ''' 
    infer a query, return it's ap

    arguments
      query       : a dict with three keys, see the template
                    {
                      'img': <path_to_img>,
                      'cls': <img class>,
                      'hist' <img histogram>
                    }
      samples     : a list of {
                                'img': <path_to_img>,
                                'cls': <img class>,
                                'hist' <img histogram>
                              }
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
  assert samples != None or (db != None and sample_db_fn != None), "need to give either samples or db plus sample_db_fn"
  if db:
    samples = sample_db_fn(db)

  q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
  results = []
  for idx, sample in enumerate(samples):
    s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
    
    if q_img == s_img:
      continue
  
    results.append({
                    'img': s_img,
                    'dis': distance(q_hist, s_hist, d_type=d_type),
                    'cls': s_cls
                  })
    
  results = sorted(results, key=lambda x: x['dis'])
  
  if depth and depth <= len(results):
    results = results[:depth]
    print(q_img)
    list_im = [sub['img'] for sub in results]
    print(list_im)
    list_im.insert(0, q_img)
    pred = [sub['cls'] for sub in results]
    weig = [sub['dis'] for sub in results]
    weig = np.reciprocal(weig)
    pred2 = weighted_mode(pred, weig)
    pred = np.array_str(pred2[0])[2:-2]
    
    list_im.insert(0, q_img)
    imgs = [PIL.Image.open(i) for i in list_im]
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    plt.imshow(imgs_comb/255.)
    plt.pause(0.1)
    plt.close()
    
  ap = myAP(q_cls, results, sort=True)

  return ap, pred

def myevaluate(db1, db2, sample_db_fn, f_class=None, f_instance=None, depth=None, d_type='d1'):
    
    samples2 = sample_db_fn(db2)
    print(len(samples2))
    
    samples1 = sample_db_fn(db1)
    print(len(samples1))
    
    classes = db1.get_class()
    classes.add(samples2[0]['cls'])
    print(classes)
    
    ret = {c: [] for c in classes}
    
    i = 0
    res = []
    
    for query in samples2:
        i+=1
        print(i)
        ap, pred = infer(query, samples=samples1, depth=depth, d_type=d_type)
        ret[query['cls']].append(ap)
        res.append(pred)
        
    return ret, res
        
def myevaluate_class(db1, db2, f_class=None, f_instance=None, depth=None, d_type='d1'):
    assert f_class or f_instance, "needs to give class_name or an"
    
    classes = db1.get_class()
    ret = {c: [] for c in classes}
    
    print('DB length', len(db1))
    print('DB length', len(db2))
    
    if f_class:
        f = f_class
    elif f_instance:
        f = f_instance
    
    samples1 = f.make_samples(db1)
    print('samples1 length', len(samples1))
    
    samples2 = f.make_samples(db2)
    print('samples2 length', len(samples2))
    
    i = 0
    
    for query in samples2:
        i+=1
        print(i)
        ap, nm = infer(query, samples=samples2, depth=depth, d_type=d_type)
        print(ap)
        print(query['cls'])
        ret[query['cls']].append(ap)
        
    return ret


def evaluate(db, sample_db_fn, depth=None, d_type='d1'):
  ''' 
    infer the whole database

    arguments
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
  classes = db.get_class()
  ret = {c: [] for c in classes}

  samples = sample_db_fn(db)
  for query in samples:
    ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
    ret[query['cls']].append(ap)

  return ret


def evaluate_class(db, f_class=None, f_instance=None, depth=None, d_type='d1'):
  ''' 
    infer the whole database

    arguments
      db     : an instance of class Database
      f_class: a class that generate features, needs to implement make_samples method
      depth  : retrieved depth during inference, the default depth is equal to database size
      d_type : distance type
  '''
  assert f_class or f_instance, "needs to give class_name or an instance of class"

  classes = db.get_class()
  ret = {c: [] for c in classes}
  
  retn = pd.DataFrame({'query': [], 'nearest1': [], 'nearest2': [], 'nearest3': []})

  if f_class:
    f = f_class()
  elif f_instance:
    f = f_instance
  samples = f.make_samples(db)
  
  i = 0
  
  for query in samples:
    ap, nm = infer(query, samples=samples, depth=depth, d_type=d_type)
    ret[query['cls']].append(ap)
    myList = [sub['img'] for sub in nm]
    retn.loc[i] = [query['img'], myList[0], myList[1], myList[2]]
    i += 1
  return ret, retn
