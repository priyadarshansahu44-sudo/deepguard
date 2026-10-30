// Real frontend JS + live FastAPI; lightweight DOM harness, not a browser render test.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
const root=path.resolve(__dirname,'..');
const elements=new Map();
function element(){return {hidden:false,disabled:false,textContent:'',value:'',style:{},children:[],handlers:{},
  classList:{toggle(){},add(){},remove(){}},addEventListener(name,fn){this.handlers[name]=fn},
  removeAttribute(){},replaceChildren(){this.children=[]},appendChild(x){this.children.push(x)},
  width:1000,height:150,getContext(){return {clearRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillRect(){}}}}}
const html=fs.readFileSync(path.join(root,'app/static/index.html'),'utf8');
for(const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1],element());
const sandbox={document:{getElementById:id=>elements.get(id),createElement:element},
  fetch:(url,options)=>fetch('http://localhost:8000'+url,options),FormData,URL,AbortSignal,console,
  testFile:new File([fs.readFileSync(path.join(root,'demo/audio_2026-10-30_18-28-14.ogg'))],'human-confirmed.ogg',{type:'audio/ogg'})};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(path.join(root,'app/static/app.js'),'utf8'),sandbox);
(async()=>{
  await vm.runInContext('health()',sandbox);
  assert.equal(elements.get('model-status').textContent,'3 trained checkpoints loaded');
  await vm.runInContext('choose(testFile)',sandbox);
  assert.equal(elements.get('analyze').disabled,false);
  await elements.get('analyze').handlers.click();
  assert.equal(elements.get('prediction-result').hidden,false);
  assert.equal(elements.get('verdict').textContent,'SYNTHETIC / DEEPFAKE');
  assert.equal(elements.get('fake-value').textContent,'70.3%');
  assert.equal(elements.get('real-value').textContent,'29.7%');
  assert.equal(elements.get('individual-predictions').children.length,3);
  const result={status:'passed',scope:'Actual frontend handlers consume live API response in a minimal DOM harness; graphical browser rendering not tested',
    verdict:elements.get('verdict').textContent,fake:elements.get('fake-value').textContent,
    genuine:elements.get('real-value').textContent,confidence:elements.get('confidence').textContent,
    individual:elements.get('individual-predictions').children.map(x=>x.textContent),ground_truth:'human, user-confirmed',correct:false};
  fs.writeFileSync(path.join(root,'reports/frontend_live_contract.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result,null,2));
})().catch(e=>{console.error(e);process.exitCode=1});
