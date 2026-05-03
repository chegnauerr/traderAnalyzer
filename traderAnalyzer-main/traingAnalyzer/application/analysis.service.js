// ═══════════════════════════════════════════════════════════════════
// STARK-OS · APPLICATION LAYER — Analysis Service
// Orquesta el motor de análisis técnico completo.
// Toda la lógica de runAnalysis + runAdvancedBrain extraída de popup.js,
// ahora con dependencias explícitas de la capa de infraestructura.
// Capa: application/analysis.service.js
// ═══════════════════════════════════════════════════════════════════

'use strict';

import { BinanceAdapter }    from '../infrastructure/binance.adapter.js';
import { IndicatorsAdapter } from '../infrastructure/indicators.adapter.js';

// Re-exportar para que popup.js pueda importar todo desde aquí
export { BinanceAdapter, IndicatorsAdapter };

// ── Alias cortos (mantiene compatibilidad con el código existente) ──────────────
const { sma: calcSMA, ema: calcEMA, rsi: calcRSI, macd: calcMACD,
        bb: calcBB, atr: calcATR, stoch: calcStoch,
        findSwings, supportResistance, detectRSIDivergence } = IndicatorsAdapter;

const { fetchKlines, fetchTicker, fetchMultiTicker,
        fetchKlinesUniversal, fetchKlinesYahoo } = BinanceAdapter;

// ── Format helpers (usados internamente en el servicio) ────────────────────────

export function fmtPrice(n) {
  if (!n || isNaN(n)) return '—';
  if (n >= 10000) return n.toLocaleString('en', { maximumFractionDigits: 0 });
  if (n >= 100)   return n.toFixed(2);
  if (n >= 1)     return n.toFixed(4);
  return n.toFixed(6);
}

// ─────────────────────────────────────────────────────────────────────────────
// HARMONIC PATTERNS
// ─────────────────────────────────────────────────────────────────────────────

function findAlternatingPivots(candles, lb = 5) {
  const raw = [];
  for (let i = lb; i < candles.length - lb; i++) {
    const hi = candles[i].high;
    const lo = candles[i].low;
    let isH = true, isL = true;
    for (let k = 1; k <= lb; k++) {
      if (candles[i-k].high >= hi || candles[i+k].high >= hi) isH = false;
      if (candles[i-k].low  <= lo || candles[i+k].low  <= lo) isL = false;
    }
    if (isH) raw.push({ i, type: 'H', price: hi });
    else if (isL) raw.push({ i, type: 'L', price: lo });
  }
  const clean = [];
  for (const p of raw) {
    const last = clean[clean.length - 1];
    if (!last || last.type !== p.type) { clean.push(p); continue; }
    if ((p.type === 'H' && p.price > last.price) || (p.type === 'L' && p.price < last.price))
      clean[clean.length - 1] = p;
  }
  return clean;
}

export function detectHarmonicPatterns(candles) {
  const price = candles[candles.length - 1].close;
  let pivots = findAlternatingPivots(candles, 5);
  if (pivots.length < 4) pivots = findAlternatingPivots(candles, 3);
  if (pivots.length < 4) return [];
  const FIB_TOL = 0.09;
  function inRange(v, lo, hi) { return v >= lo * (1 - FIB_TOL) && v <= hi * (1 + FIB_TOL); }
  const DEFS = [
    { name:'Gartley',   abMin:0.58, abMax:0.66, bcMin:0.38, bcMax:0.89, dMin:0.74, dMax:0.83, acc:78 },
    { name:'Bat',       abMin:0.36, abMax:0.52, bcMin:0.38, bcMax:0.89, dMin:0.84, dMax:0.92, acc:82 },
    { name:'Butterfly', abMin:0.74, abMax:0.84, bcMin:0.38, bcMax:0.89, dMin:1.22, dMax:1.70, acc:73 },
    { name:'Crab',      abMin:0.36, abMax:0.62, bcMin:0.38, bcMax:0.89, dMin:1.55, dMax:1.68, acc:71 },
    { name:'Shark',     abMin:0.44, abMax:0.56, bcMin:1.00, bcMax:1.68, dMin:0.85, dMax:0.92, acc:68 },
  ];
  const results = [];
  const recent = pivots.slice(-8);
  for (let i = 0; i <= recent.length - 4; i++) {
    const [X, A, B, C] = recent.slice(i, i + 4);
    if (X.type === A.type || A.type === B.type || B.type === C.type) continue;
    const XA = Math.abs(A.price - X.price);
    const AB = Math.abs(B.price - A.price);
    const BC = Math.abs(C.price - B.price);
    if (XA < 1e-12 || AB < 1e-12) continue;
    const abXa = AB / XA;
    const bcAb = BC / AB;
    const isBull = A.type === 'L';
    for (const def of DEFS) {
      if (!inRange(abXa, def.abMin, def.abMax)) continue;
      if (!inRange(bcAb, def.bcMin, def.bcMax)) continue;
      const dir = A.price > X.price ? 1 : -1;
      const dMid  = X.price - dir * XA * ((def.dMin + def.dMax) / 2);
      const dLow  = X.price - dir * XA * def.dMax;
      const dHigh = X.price - dir * XA * def.dMin;
      const distPct = Math.abs(price - dMid) / Math.max(price, dMid) * 100;
      let stage, prob;
      if (distPct < 1.5)  { stage='COMPLETANDO';       prob=def.acc; }
      else if (distPct<5) { stage='APPROACHANDO PRZ';  prob=Math.round(def.acc*0.8); }
      else                { stage='FORMANDO';           prob=Math.round(def.acc*0.5); }
      const fib618 = dMid + (isBull?1:-1)*Math.abs(C.price-dMid)*0.618;
      results.push({
        name:def.name, type:isBull?'bullish':'bearish', stage, prob,
        dZone:{low:Math.min(dLow,dHigh),high:Math.max(dLow,dHigh),mid:dMid},
        sl:   isBull?Math.min(dLow,dHigh)*0.993:Math.max(dLow,dHigh)*1.007,
        tp1:fib618, tp2:isBull?A.price:A.price,
        distPct:distPct.toFixed(1),
        ratios:{'AB/XA':abXa.toFixed(3),'BC/AB':bcAb.toFixed(3)},
      });
      break;
    }
  }
  const seen=new Set();
  return results.sort((a,b)=>b.prob-a.prob)
    .filter(p=>{if(seen.has(p.name))return false;seen.add(p.name);return true;}).slice(0,4);
}

// ─────────────────────────────────────────────────────────────────────────────
// ELLIOTT WAVE
// ─────────────────────────────────────────────────────────────────────────────

export function detectElliottWave(candles) {
  const price = candles[candles.length - 1].close;
  let pivots = findAlternatingPivots(candles, 4);
  if (pivots.length < 5) pivots = findAlternatingPivots(candles, 3);
  if (pivots.length < 5) pivots = findAlternatingPivots(candles, 2);
  if (pivots.length < 5) return null;
  const recent = pivots.slice(-7);
  for (let s = 0; s <= recent.length - 5; s++) {
    const [p0,p1,p2,p3,p4] = recent.slice(s,s+5);
    const isBull = p0.type==='L'&&p1.type==='H'&&p2.type==='L'&&p3.type==='H'&&p4.type==='L';
    const isBear = p0.type==='H'&&p1.type==='L'&&p2.type==='H'&&p3.type==='L'&&p4.type==='H';
    if (!isBull && !isBear) continue;
    const w1=Math.abs(p1.price-p0.price),w2=Math.abs(p2.price-p1.price);
    const w3=Math.abs(p3.price-p2.price),w4=Math.abs(p4.price-p3.price);
    if (!w1||!w3) continue;
    const hard1=w2/w1<1.0, hard2=w4/w3<1.0;
    if (!hard1||!hard2) continue;
    const soft1=w3>=Math.min(w1,w4);
    const soft2=isBull?p4.price>p1.price:p4.price<p1.price;
    if (!soft1&&!soft2) continue;
    const score=[hard1,hard2,soft1,soft2].filter(Boolean).length;
    const dir=isBull?'bullish':'bearish';
    const confidence=45+score*13;
    let currentWave,nextMove,nextTarget;
    if(isBull){
      if(price>p3.price){currentWave='W5 ▲';nextMove='Impulso final alcista — buscar techo';nextTarget=p3.price+w3*0.618;}
      else if(price>p2.price){currentWave='W4 ↘ (correctivo)';nextMove='Corrección — soporte en ~'+fmtPrice(p2.price)+', luego W5';nextTarget=p3.price+w1*0.618;}
      else{currentWave='W3 ▲ (más fuerte)';nextMove='Impulso W3 en marcha — la más poderosa';nextTarget=p1.price+w1*1.618;}
    } else {
      if(price<p3.price){currentWave='W5 ▼';nextMove='Impulso bajista final — buscar suelo';nextTarget=p3.price-w3*0.618;}
      else if(price<p2.price){currentWave='W4 ↗ (correctivo)';nextMove='Corrección — resistencia en ~'+fmtPrice(p2.price)+', luego W5 bajista';nextTarget=p3.price-w1*0.618;}
      else{currentWave='W3 ▼ (más fuerte)';nextMove='Caída W3 en marcha';nextTarget=p1.price-w1*1.618;}
    }
    return {type:'impulso',dir,currentWave,nextMove,nextTarget,
      confidence:Math.min(88,confidence),
      w2Ret:(w2/w1*100).toFixed(0),w3vsW1:(w3/w1).toFixed(2),w4Ret:(w4/w3*100).toFixed(0),
      pts:{p0,p1,p2,p3,p4}};
  }
  if(recent.length>=3){
    const [pA,pB,pC]=recent.slice(-3);
    if(pA.type!==pB.type&&pB.type!==pC.type){
      const wA=Math.abs(pB.price-pA.price),wB=Math.abs(pC.price-pB.price);
      const bcRet=wB/wA;
      if(bcRet<1.0){
        const dir=pA.type==='H'?'bearish':'bullish';
        const cTarget=pA.type==='H'?pB.price-wA*0.618:pB.price+wA*0.618;
        return{type:'corrección',dir,currentWave:'Wave C',confidence:52,
          nextMove:`Wave C ${dir==='bearish'?'bajista':'alcista'} — objetivo ~$${fmtPrice(cTarget)}`,
          nextTarget:cTarget,w2Ret:null,w3vsW1:null,w4Ret:null,pts:{p0:pA,p1:pB,p2:pC}};
      }
    }
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// MULTI-TIMEFRAME
// ─────────────────────────────────────────────────────────────────────────────

const MTF_MAP = {
  '1m':  {primary:'1h',  secondary:'15m'},
  '5m':  {primary:'4h',  secondary:'1h'},
  '15m': {primary:'4h',  secondary:'1h'},
  '30m': {primary:'1d',  secondary:'4h'},
  '1h':  {primary:'1d',  secondary:'4h'},
  '4h':  {primary:'1w',  secondary:'1d'},
  '1d':  {primary:'1w',  secondary:'4h'},
  '1w':  {primary:'1w',  secondary:'1d'},
};

export async function fetchMTFanalysis(symbol, currentTf) {
  const tfConf = MTF_MAP[currentTf] || {primary:'1d', secondary:'4h'};
  const calcSR = (cArr) => {
    if (!cArr||cArr.length<20) return {res:null,sup:null};
    const price=cArr[cArr.length-1].close;
    const lb=4; let res=null,sup=null;
    for(let i=lb;i<cArr.length-lb;i++){
      const sl=cArr.slice(i-lb,i+lb+1);
      if(cArr[i].high>=Math.max(...sl.map(c=>c.high))&&cArr[i].high>price){if(!res||cArr[i].high<res)res=cArr[i].high;}
      if(cArr[i].low<=Math.min(...sl.map(c=>c.low))&&cArr[i].low<price){if(!sup||cArr[i].low>sup)sup=cArr[i].low;}
    }
    return {res,sup};
  };
  const calcTrend=(closes,cArr)=>{
    if(!closes?.length) return null;
    const n=closes.length;
    const ema=(arr,per)=>{const k=2/(per+1);return arr.reduce((prev,v,i)=>i===0?v:prev*(1-k)+v*k);};
    const ema20=ema(closes.slice(-20),20),ema50=ema(closes.slice(-Math.min(50,n)),50);
    const last=closes[n-1];
    const rsiData=closes.slice(-15);
    const gains=[],losses=[];
    for(let i=1;i<rsiData.length;i++){const d=rsiData[i]-rsiData[i-1];gains.push(d>0?d:0);losses.push(d<0?-d:0);}
    const ag=gains.reduce((a,b)=>a+b,0)/gains.length,al=losses.reduce((a,b)=>a+b,0)/losses.length;
    const rsi=al===0?99:Math.round(100-100/(1+ag/al));
    let dir;
    if(last>ema20&&ema20>ema50)dir='bullish';
    else if(last<ema20&&ema20<ema50)dir='bearish';
    else if(last>ema20)dir='neutral_up';
    else dir='neutral_down';
    const sr=calcSR(cArr);
    return{dir,rsi,ema20:ema20.toFixed(2),last:last.toFixed(2),res:sr.res,sup:sr.sup};
  };
  try {
    const [primCandles,secCandles]=await Promise.all([
      fetchKlinesUniversal(symbol,tfConf.primary,80).catch(()=>null),
      fetchKlinesUniversal(symbol,tfConf.secondary,80).catch(()=>null),
    ]);
    const prim=calcTrend(primCandles?.map(c=>c.close),primCandles);
    const sec =calcTrend(secCandles?.map(c=>c.close),secCandles);
    if(!prim&&!sec) return null;
    const dirs=[prim?.dir,sec?.dir].filter(Boolean);
    const bullN=dirs.filter(d=>d.includes('bullish')||d==='neutral_up').length;
    const bearN=dirs.filter(d=>d.includes('bearish')||d==='neutral_down').length;
    let align,alignScore;
    if(bullN===2){align='ALCISTA ↑↑';alignScore=88;}
    else if(bearN===2){align='BAJISTA ↓↓';alignScore=88;}
    else if(bullN>bearN){align='ALCISTA ↑~';alignScore=62;}
    else if(bearN>bullN){align='BAJISTA ↓~';alignScore=62;}
    else{align='DIVERGENTE ↕';alignScore=35;}
    return{primaryTf:tfConf.primary,secondaryTf:tfConf.secondary,currentTf,prim,sec,align,alignScore};
  } catch {return null;}
}

// ─────────────────────────────────────────────────────────────────────────────
// ADVANCED PROBABILITY ENGINE
// ─────────────────────────────────────────────────────────────────────────────

export function computeAdvancedProb({score,mtf,harmonics,elliott,div,volume,patterns,mlResult}) {
  let bull,bear,conf;
  const factors=[];
  if(mlResult){
    bull=mlResult.bullPct*0.6+score*0.4;
    bear=mlResult.bearPct*0.6+(100-score)*0.4;
    conf=50+Math.round((mlResult.auc-0.5)*80);
    factors.push(`🤖 ML ${mlResult.bullPct}% alcista · AUC ${mlResult.auc}`);
  } else {
    bull=score;bear=100-score;conf=48;
    factors.push('Reglas técnicas (sin modelo ML)');
  }
  if(mtf){
    if(mtf.align==='ALCISTA ↑↑'){bull+=18;conf+=18;factors.push('MTF ↑↑ alineado alcista');}
    else if(mtf.align==='BAJISTA ↓↓'){bear+=18;conf+=18;factors.push('MTF ↓↓ alineado bajista');}
    else if(mtf.align?.includes('ALCISTA')){bull+=9;conf+=6;factors.push('MTF parcialmente alcista');}
    else if(mtf.align?.includes('BAJISTA')){bear+=9;conf+=6;factors.push('MTF parcialmente bajista');}
    else{conf-=12;factors.push('MTF divergente — señal reducida');}
  }
  const bestH=harmonics?.[0];
  if(bestH?.stage==='COMPLETANDO'||bestH?.stage==='APPROACHANDO PRZ'){
    const w=bestH.stage==='COMPLETANDO'?22:14;
    if(bestH.type==='bullish'){bull+=w;conf+=14;}else{bear+=w;conf+=14;}
    factors.push(`Armónico ${bestH.name} ${bestH.stage} (${bestH.prob}%)`);
  }
  if(elliott){
    const w=elliott.currentWave;
    if(w?.includes('W3')&&elliott.dir==='bullish'){bull+=16;conf+=10;factors.push(`Elliott ${w} — ola más fuerte`);}
    else if(w?.includes('W3')&&elliott.dir==='bearish'){bear+=16;conf+=10;factors.push(`Elliott ${w}`);}
    else if(w?.includes('W5')&&elliott.dir==='bullish'){bull+=8;bear+=4;factors.push(`Elliott ${w} — ola final`);}
    else if(w?.includes('W5')&&elliott.dir==='bearish'){bear+=8;bull+=4;factors.push(`Elliott ${w} — ola final`);}
    else if(w?.includes('W4')){conf-=8;factors.push(`Elliott ${w} — corrección activa`);}
    conf+=Math.round(elliott.confidence*0.12);
  }
  if(div?.type==='bullish'){bull+=14;conf+=8;factors.push('Divergencia alcista RSI');}
  if(div?.type==='bearish'){bear+=14;conf+=8;factors.push('Divergencia bajista RSI');}
  if(volume?.spike){
    if(volume.bias==='buyPressure'){bull+=8;factors.push(`Spike volumen ${volume.ratio?.toFixed(1)}x comprador`);}
    else{bear+=8;factors.push(`Spike volumen ${volume.ratio?.toFixed(1)}x vendedor`);}
  }
  const topPat=patterns?.find(p=>p.probability>=60);
  if(topPat){
    if(topPat.type==='bullish'){bull+=10;factors.push(`Patrón ${topPat.name} (${topPat.probability}%)`);}
    else{bear+=10;factors.push(`Patrón ${topPat.name} (${topPat.probability}%)`);}
  }
  const total=bull+bear||1;
  const bullPct=Math.min(97,Math.max(3,Math.round(bull/total*100)));
  const bearPct=100-bullPct;
  conf=Math.min(96,Math.max(28,Math.round(conf)));
  return{bullPct,bearPct,conf,factors};
}

// ─────────────────────────────────────────────────────────────────────────────
// SINGLETON EXPORT — AnalysisService
// ─────────────────────────────────────────────────────────────────────────────

export const AnalysisService = Object.freeze({
  detectHarmonicPatterns,
  detectElliottWave,
  fetchMTFanalysis,
  computeAdvancedProb,
  fmtPrice,
  // Indicadores (re-expuestos para popup.js)
  calcSMA, calcEMA, calcRSI, calcMACD, calcBB, calcATR, calcStoch,
  findSwings, supportResistance, detectRSIDivergence,
  // Market data (re-expuestos para popup.js)
  fetchKlines, fetchTicker, fetchMultiTicker,
  fetchKlinesUniversal, fetchKlinesYahoo,
});

export default AnalysisService;
