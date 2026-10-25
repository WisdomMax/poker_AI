const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
function setup(){
 const sounds=[];
 const context={localStorage:{getItem:()=>null,setItem(){}},document:{addEventListener(){}},window:{addEventListener(){},removeEventListener(){}},console,setTimeout,clearTimeout,Date,Audio:class{constructor(url){this.url=url;this.paused=false;sounds.push(this)} play(){return Promise.resolve()} pause(){this.paused=true}}};
 vm.runInNewContext(fs.readFileSync(__dirname+'/static/js/audio_manager.js','utf8'),context);
 const manager=context.window.arenaAudio;manager.unlocked=true;manager.setCurrentHand(1);
 return {manager,sounds};
}
test('old playback completion cannot resume after a new hand', async()=>{
 const {manager:m,sounds}=setup();
 m.enqueue('old','voice',1);m.enqueue('pending','voice',1);
 m.setCurrentHand(2);m.enqueue('new','voice',2);
 await new Promise(setImmediate);
 assert.equal(m.currentAudio.url,'new');assert.equal(m.isPlaying,true);
 assert.equal(sounds[0].paused,true);assert.equal(m.audioQueue.length,0);
 m.stopAllAndClear();
});
test('pending speech is bounded, stale hands discarded, effects stop on mute', async()=>{
 const {manager:m,sounds}=setup();
 m.enqueue('first','voice',1);m.enqueue('second','voice',1);m.enqueue('latest','voice',1);
 assert.equal(m.audioQueue.length,1);assert.equal(m.audioQueue[0].url,'latest');
 m.playSfx('chip_call');m.toggleMute();await new Promise(setImmediate);
 assert(sounds.every(a=>a.paused));assert.equal(m.isPlaying,false);
 m.toggleMute();m.enqueue('stale','voice',0);m.stopAllAndClear();
});
