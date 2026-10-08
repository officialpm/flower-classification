# Copyright 2026 Parth Maniar. Licensed under Apache-2.0.
import os,glob,time,json,hashlib,platform
from IPython.display import display
from pathlib import Path
import numpy as np,pandas as pd,tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score,log_loss,classification_report
SEED=2026;N_CLASSES=104;SIZE=224;OUT=Path(os.environ.get('FLOWER_OUTPUT_DIR','outputs'));OUT.mkdir(parents=True,exist_ok=True);START=time.time()
tf.keras.utils.set_random_seed(SEED)
resolver=None
gpus=tf.config.list_physical_devices('GPU')
if not gpus:raise RuntimeError('GPU required for the measured workflow; CPU training is disabled.')
strategy=tf.distribute.MirroredStrategy()
BATCH=16*strategy.num_replicas_in_sync
print({'tensorflow':tf.__version__,'replicas':strategy.num_replicas_in_sync,'batch':BATCH,'seed':SEED})

roots=list(Path(os.environ.get('FLOWER_DATA_DIR','data')).rglob('tfrecords-jpeg-224x224'))
assert len(roots)==1, f'Expected one 224px competition root, found {roots}'
local=roots[0]
base=str(local)
files={s:sorted(tf.io.gfile.glob(base+'/'+s+'/*.tfrec')) for s in ['train','val','test']}
assert all(files.values()),files
raw=next(iter(tf.data.TFRecordDataset(files['train'][:1])))
example=tf.train.Example.FromString(bytes(raw.numpy()))
keys=set(example.features.feature)
IMAGE_KEY='image' if 'image' in keys else 'img'
LABEL_KEY='class' if 'class' in keys else 'label'
assert IMAGE_KEY in keys and LABEL_KEY in keys, keys
samples=list(Path(os.environ.get('FLOWER_DATA_DIR','data')).rglob('sample_submission.csv'))
assert len(samples)==1,samples
sample=pd.read_csv(samples[0],dtype={'id':str})
assert list(sample.columns)==['id','label'] and sample.id.is_unique
print('Schema:',keys,'test sample rows:',len(sample))
AUTO=tf.data.AUTOTUNE

def parse(record,labeled):
    schema={IMAGE_KEY:tf.io.FixedLenFeature([],tf.string)}
    schema[LABEL_KEY if labeled else 'id']=tf.io.FixedLenFeature([],tf.int64 if labeled else tf.string)
    v=tf.io.parse_single_example(record,schema)
    # EfficientNet contains its own normalization; input remains [0,255].
    im=tf.cast(tf.io.decode_jpeg(v[IMAGE_KEY],channels=3),tf.float32)
    im=tf.ensure_shape(im,[SIZE,SIZE,3])
    return im,tf.cast(v[LABEL_KEY],tf.int32) if labeled else v['id']

def dataset(split):
    d=tf.data.TFRecordDataset(files[split],num_parallel_reads=1)
    opt=tf.data.Options();opt.experimental_deterministic=True
    return d.with_options(opt).map(lambda r:parse(r,split!='test'),num_parallel_calls=AUTO)
train=dataset('train');val=dataset('val');test=dataset('test')
ytrain=np.concatenate([y.numpy() for _,y in train.batch(256)])
yval=np.concatenate([y.numpy() for _,y in val.batch(256)])
assert set(ytrain)==set(range(N_CLASSES)) and set(yval)==set(range(N_CLASSES))
assert ytrain.min()>=0 and ytrain.max()<N_CLASSES
assert yval.min()>=0 and yval.max()<N_CLASSES
print('Observed counts:',len(ytrain),len(yval),len(sample))
fig,ax=plt.subplots(figsize=(12,3));ax.bar(np.arange(N_CLASSES),np.bincount(ytrain,minlength=N_CLASSES));ax.set(xlabel='Class ID',ylabel='Training rows',title='Class coverage');plt.tight_layout();plt.savefig(OUT/"validation_comparison.png");plt.close()
train_batch=train.shuffle(4096,seed=SEED,reshuffle_each_iteration=True).batch(BATCH).prefetch(AUTO)
val_batch=val.batch(BATCH).prefetch(AUTO)

with strategy.scope():
    inputs=tf.keras.Input((SIZE,SIZE,3))
    aug=tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(.08),tf.keras.layers.RandomZoom(.1)])
    backbone=tf.keras.applications.EfficientNetB0(include_top=False,weights='imagenet',input_shape=(SIZE,SIZE,3))
    backbone.trainable=False
    z=backbone(aug(inputs),training=False)
    z=tf.keras.layers.GlobalAveragePooling2D()(z);z=tf.keras.layers.Dropout(.25)(z)
    outputs=tf.keras.layers.Dense(N_CLASSES,activation='softmax',dtype='float32')(z)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
class TimeBudget(tf.keras.callbacks.Callback):
    def on_train_batch_end(self,batch,logs=None):
        if time.time()-START>7800:
            self.model.stop_training=True
    def on_epoch_end(self,epoch,logs=None):
        if time.time()-START>7800:
            self.model.stop_training=True
def cb(path):
    return [TimeBudget(),tf.keras.callbacks.ModelCheckpoint(str(path),monitor='val_loss',save_best_only=True,save_weights_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)]
head_path=OUT/'head.weights.h5'
hhead=model.fit(train_batch,validation_data=val_batch,epochs=3,callbacks=cb(head_path))
model.load_weights(head_path)
head_val=model.predict(val_batch.map(lambda x,y:x),verbose=0)
head_test=model.predict(test.batch(BATCH).map(lambda x,i:x).prefetch(AUTO),verbose=0)
np.save(OUT/'head_val.npy',head_val);np.save(OUT/'head_test.npy',head_test)
if time.time()-START>7200: raise RuntimeError('Two-hour guard: stop before the three-hour competition cap.')
with strategy.scope():
    backbone.trainable=True
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
fine_path=OUT/'fine.weights.h5'
hfine=model.fit(train_batch,validation_data=val_batch,epochs=8,callbacks=cb(fine_path))
model.load_weights(fine_path)
fine_val=model.predict(val_batch.map(lambda x,y:x),verbose=0)
fine_test=model.predict(test.batch(BATCH).map(lambda x,i:x).prefetch(AUTO),verbose=0)
np.save(OUT/'fine_val.npy',fine_val);np.save(OUT/'fine_test.npy',fine_test)

candidates={'head':(head_val,head_test),'fine':(fine_val,fine_test),'equal_blend':((head_val+fine_val)/2,(head_test+fine_test)/2)}
records=[]
for name,(pv,pt) in candidates.items():
    assert pv.shape==(len(yval),N_CLASSES) and pt.shape==(len(sample),N_CLASSES)
    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    assert np.allclose(pv.sum(1),1,atol=1e-4) and np.allclose(pt.sum(1),1,atol=1e-4)
    records.append({'candidate':name,'macro_f1':f1_score(yval,pv.argmax(1),labels=range(N_CLASSES),average='macro',zero_division=0),'accuracy':accuracy_score(yval,pv.argmax(1)),'logloss':log_loss(yval,pv,labels=range(N_CLASSES))})
metrics=pd.DataFrame(records).sort_values('macro_f1',ascending=False);display(metrics)
majority=np.full(len(yval),np.bincount(ytrain).argmax());print('Majority macro-F1:',f1_score(yval,majority,labels=range(N_CLASSES),average='macro',zero_division=0))
best=metrics.iloc[0].candidate;pv,pt=candidates[best]
report=pd.DataFrame(classification_report(yval,pv.argmax(1),labels=range(N_CLASSES),output_dict=True,zero_division=0)).T
report.to_csv(OUT/'validation_class_metrics.csv');metrics.to_csv(OUT/'validation_candidates.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4));metrics.plot.bar(x='candidate',y='macro_f1',ax=axes[0],legend=False);axes[0].set_title('Official validation macro-F1');report.iloc[:N_CLASSES]['f1-score'].plot.bar(ax=axes[1]);axes[1].set_xticks([]);axes[1].set_title('Per-class validation F1');plt.tight_layout();plt.savefig(OUT/"validation_comparison.png");plt.close()

test_ids=np.concatenate([ids.numpy() for _,ids in test.batch(256)]).astype(str)
assert len(test_ids)==len(sample) and len(set(test_ids))==len(test_ids)
assert set(test_ids)==set(sample.id)
preds=pd.DataFrame({'id':test_ids,'label':pt.argmax(1)})
submission=sample[['id']].merge(preds,on='id',how='left',validate='one_to_one',sort=False)
assert submission.id.tolist()==sample.id.tolist() and submission.label.notna().all()
submission.label=submission.label.astype(int)
assert submission.label.between(0,N_CLASSES-1).all()
submission.to_csv(OUT/'submission.csv',index=False)
sha=hashlib.sha256((OUT/'submission.csv').read_bytes()).hexdigest()
manifest={'selected':best,'validation':metrics.iloc[0].to_dict(),'public_score':None,'test_rows':len(sample),'train_rows':len(ytrain),'validation_rows':len(yval),'seed':SEED,'image_size':SIZE,'tensorflow':tf.__version__,'python':platform.python_version(),'elapsed_seconds':time.time()-START,'pretraining':'Keras EfficientNetB0 ImageNet','test_labels_accessed':False,'sha256':sha,'copyright':'Copyright 2026 Parth Maniar. Apache-2.0.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2));display(submission.head())
