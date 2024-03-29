import {isInViewport} from "./util.js"

export function mutatingGCADemo() {
  const $ = q=>document.querySelector(q);

  const sleep = (ms)=>new Promise(resolve => setTimeout(resolve, ms));

  const parseConsts = model_graph=>{
    const dtypes = {'DT_INT32':['int32', 'intVal', Int32Array],
                    'DT_FLOAT':['float32', 'floatVal', Float32Array]};
    
    const consts = {};
    model_graph.modelTopology.node.filter(n=>n.op=='Const').forEach((node=>{
      const v = node.attr.value.tensor;
      const [dtype, field, arrayType] = dtypes[v.dtype];
      if (!v.tensorShape.dim) {
        consts[node.name] = [tf.scalar(v[field][0], dtype)];
      } else {
        const shape = v.tensorShape.dim.map(d=>parseInt(d.size));
        let arr;
        if (v.tensorContent) {
          const data = atob(v.tensorContent);
          const buf = new Uint8Array(data.length);
          for (var i=0; i<data.length; ++i) {
            buf[i] = data.charCodeAt(i);
          }
          arr = new arrayType(buf.buffer);
        } else {
          const size = shape.reduce((a, b)=>a*b);
          arr = new arrayType(size);
          arr.fill(v[field][0]);
        }
        consts[node.name] = [tf.tensor(arr, shape, dtype)];
      }
    }));
    return consts;
  }

  let kTail = 0.0;
  let kLeg = 0.0;
  let kHead = 0.0;
  let kArm = 0.0;
  let kRed = 0.0;
  let kBlue = 0.0;

  let forcesum1Ckbx = document.getElementById("forcesum1");
  let tailSlider = document.getElementById("tailSlider");
  let legSlider = document.getElementById("legSlider");
  let headSlider = document.getElementById("headSlider");
  let armSlider = document.getElementById("armSlider");
  let redSlider = document.getElementById("redSlider");
  let blueSlider = document.getElementById("blueSlider");


  const run = async ()=>{

    let perturbationMatrix = tf.eye(16);

    const perturbations = tf.tensor(PERTURBATIONS);
    const tailPertM = perturbations.gather([0]).squeeze();
    const legPertM = perturbations.gather([1]).squeeze();
    const headPertM = perturbations.gather([2]).squeeze();
    const armPertM = perturbations.gather([3]).squeeze();
    const redPertM = perturbations.gather([4]).squeeze();
    const bluePertM = perturbations.gather([5]).squeeze();

    const I = tf.eye(16);
    const updatePerturbation = () => {
        let kI = 1.0 - kTail - kLeg - kHead - kArm - kRed - kBlue;
        perturbationMatrix = I.mul(kI).
          add(tailPertM.mul(kTail)).
          add(legPertM.mul(kLeg)).
          add(headPertM.mul(kHead)).
          add(armPertM.mul(kArm)).
          add(redPertM.mul(kRed)).
          add(bluePertM.mul(kBlue));
    }


    $('#tailSlider').oninput = (e) =>{
        updateK("tail", parseFloat(e.target.value));
    }
    $('#legSlider').oninput = (e) =>{
        updateK("leg", parseFloat(e.target.value));
    }
    $('#headSlider').oninput = (e) =>{
        updateK("head", parseFloat(e.target.value));
    }
    $('#armSlider').oninput = (e) =>{
        updateK("arm", parseFloat(e.target.value));
    }
    $('#redSlider').oninput = (e) =>{
        updateK("red", parseFloat(e.target.value));
    }
    $('#blueSlider').oninput = (e) =>{
        updateK("blue", parseFloat(e.target.value));
    }

    const updateKUnchecked = (kid, v) => {
      if (kid == "tail"){
        kTail = v;
        $('#tailPerturbation').innerText = kTail.toFixed(2);
      } else if (kid == "leg"){
        kLeg = v;
        $('#legPerturbation').innerText = kLeg.toFixed(2);
      } else if (kid == "head"){
        kHead = v;
        $('#headPerturbation').innerText = kHead.toFixed(2);
      } else if (kid == "arm"){
        kArm = v;
        $('#armPerturbation').innerText = kArm.toFixed(2);
      } else if (kid == "red"){
        kRed = v;
        $('#redPerturbation').innerText = kRed.toFixed(2);
      } else if (kid == "blue"){
        kBlue = v;
        $('#bluePerturbation').innerText = kBlue.toFixed(2);
      } else {
        console.log("ERROR!");
      }
    }


    const updateK = (kid, v) => {
        if (forcesum1Ckbx.checked == false) {
          updateKUnchecked(kid, v);
        } else {
          // You cannot go over 1.
          let vAbs = Math.abs(v);
          const vSign = Math.sign(v);
          const kTailAbs = Math.abs(kTail);
          const kTailSign = Math.sign(kTail);
          const kLegAbs = Math.abs(kLeg);
          const kLegSign = Math.sign(kLeg);
          const kHeadAbs = Math.abs(kHead);
          const kHeadSign = Math.sign(kHead);
          const kArmAbs = Math.abs(kArm);
          const kArmSign = Math.sign(kArm);
          const kRedAbs = Math.abs(kRed);
          const kRedSign = Math.sign(kRed);
          const kBlueAbs = Math.abs(kBlue);
          const kBlueSign = Math.sign(kBlue);

          let kCurrAbs;
          if (kid == "tail"){
            kCurrAbs = kTailAbs;
          } else if (kid == "leg"){
            kCurrAbs = kLegAbs;
          } else if (kid == "head"){
            kCurrAbs = kHeadAbs;
          } else if (kid == "arm"){
            kCurrAbs = kArmAbs;
          } else if (kid == "red"){
            kCurrAbs = kRedAbs;
          } else if (kid == "blue"){
            kCurrAbs = kBlueAbs;
          }
          let totK = vAbs + kTailAbs + kLegAbs + kHeadAbs + kArmAbs
                + kRedAbs + kBlueAbs - kCurrAbs;
          if (totK <= 1.0) {
            // No problem here, do just like you did before.
            updateKUnchecked(kid, v);
          } else {
            // Prevent v from going over 1.
            if (vAbs > 1.0) {
              vAbs = 1.0;
              totK = vAbs + kTailAbs + kLegAbs + kHeadAbs + kArmAbs
                + kRedAbs + kBlueAbs - kCurrAbs;
            }
            // Subtract the excess from the rest.
            const excess = totK - 1.0;

            const tailContrib = kid == "tail" ? 0.0 : kTailAbs;
            const legContrib = kid == "leg" ? 0.0 : kLegAbs;
            const headContrib = kid == "head" ? 0.0 : kHeadAbs;
            const armContrib = kid == "arm" ? 0.0 : kArmAbs;
            const redContrib = kid == "red" ? 0.0 : kRedAbs;
            const blueContrib = kid == "blue" ? 0.0 : kBlueAbs;
            const totContrib = tailContrib + legContrib + headContrib +
                armContrib + redContrib + blueContrib;

            let tailDecr = 0.0;
            let legDecr = 0.0;
            let headDecr = 0.0;
            let armDecr = 0.0;
            let redDecr = 0.0;
            let blueDecr = 0.0;
            if (totContrib > 1e-6) {
              tailDecr = tailContrib / totContrib * excess;
              legDecr = legContrib / totContrib * excess;
              headDecr = headContrib / totContrib * excess;
              armDecr = armContrib / totContrib * excess;
              redDecr = redContrib / totContrib * excess;
              blueDecr = blueContrib / totContrib * excess;
            }

            kTail = kid == "tail" ? vAbs * vSign : kTailSign * (kTailAbs - tailDecr);
            kLeg = kid == "leg" ? vAbs * vSign : kLegSign * (kLegAbs - legDecr);
            kHead = kid == "head" ? vAbs * vSign : kHeadSign * (kHeadAbs - headDecr);
            kArm = kid == "arm" ? vAbs * vSign : kArmSign * (kArmAbs - armDecr);
            kRed = kid == "red" ? vAbs * vSign : kRedSign * (kRedAbs - redDecr);
            kBlue = kid == "blue" ? vAbs * vSign : kBlueSign * (kBlueAbs - blueDecr);
            $('#tailPerturbation').innerText = kTail.toFixed(2);
            tailSlider.value = kTail;
            $('#legPerturbation').innerText = kLeg.toFixed(2);
            legSlider.value = kLeg;
            $('#headPerturbation').innerText = kHead.toFixed(2);
            headSlider.value = kHead;
            $('#armPerturbation').innerText = kArm.toFixed(2);
            armSlider.value = kArm;
            $('#redPerturbation').innerText = kRed.toFixed(2);
            redSlider.value = kRed;
            $('#bluePerturbation').innerText = kBlue.toFixed(2);
            blueSlider.value = kBlue;
          }
        }
        updatePerturbation();
    }


    const r = await fetch(GRAPH_URL);
    const consts = parseConsts(await r.json());
    
    const model = await tf.loadGraphModel(GRAPH_URL);
    Object.assign(model.weights, consts);

    
    let seed = new Array(16).fill(0).map((x, i)=>i<3?0:1);
    seed = tf.tensor(seed, [1, 1, 1, 16]);
    
    const D = 96;
    const initState = tf.tidy(()=>{
      const D2 = D/2;
      const a = seed.pad([[0, 0], [D2-1, D2], [D2-1, D2], [0,0]]);
      return a;
    });
    
    const state = tf.variable(initState);
    const [_, h, w, ch] = state.shape;
    

    $('#reset').onclick = (e)=>{
        tf.tidy(()=>{
          state.assign(initState);
        });
    }


    const damage = (x, y, r)=>{
      tf.tidy(()=>{
        const rx = tf.range(0, w).sub(x).div(r).square().expandDims(0);
        const ry = tf.range(0, h).sub(y).div(r).square().expandDims(1);
        const mask = rx.add(ry).greater(1.0).expandDims(2);
        state.assign(state.mul(mask));
      });
    }
    
    const plantSeed = (x, y)=>{
      const x2 = w-x-seed.shape[2];
      const y2 = h-y-seed.shape[1];
      if (x<0 || x2<0 || y2<0 || y2<0)
        return;
      tf.tidy(()=>{
        const a = seed.pad([[0, 0], [y, y2], [x, x2], [0,0]]);
        state.assign(state.add(a));
      });
    }
    
    const scale = 4;
    
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = w;
    canvas.height = h;
    // canvas.style.width = `${w*scale}px`;
    // canvas.style.height = `${h*scale}px`;
    
    function canvasToGrid(xin, yin) {
          const x = xin / canvas.clientWidth * w;
          const y = yin / canvas.clientHeight * h;
          return [x, y];    
      }
    const getClickPos = e=>{
      return canvasToGrid(e.offsetX, e.offsetY);
    }
    function getTouchPos(touch) {
      const rect = canvas.getBoundingClientRect();
      return canvasToGrid(touch.clientX - rect.left, touch.clientY - rect.top);
    }


    canvas.onmousedown = e=>{
      const [x, y] = getClickPos(e);
        if (e.buttons == 1) {
          if (e.shiftKey) {
            plantSeed(x, y);  
          } else {
            damage(x, y, 8);
          }
        }
    }
    canvas.onmousemove = e=>{
      const [x, y] = getClickPos(e);
      if (e.buttons == 1 && !e.shiftKey) {
        damage(x, y, 8);
      }
    }

    let lastTouchList = null;

    canvas.addEventListener("touchstart", e => {
        e.preventDefault();
        for (const t of e.changedTouches) {
            const [x, y] = getTouchPos(t);
            damage(x,y, 8);
        }
    });

    canvas.addEventListener("touchmove", e => {
        e.preventDefault();
        for (const t of e.changedTouches) {
            const [x, y] = getTouchPos(t);
            damage(x, y, 8);
        }
    });

    function step() {
      tf.tidy(()=>{
        let new_state = model.execute(
            {x:state, fire_rate:tf.tensor(0.5),
            angle:tf.tensor(0.0), step_size:tf.tensor(1.0)}, ['Identity']);
        new_state = new_state.reshape([-1, 16]);
        new_state = new_state.matMul(perturbationMatrix).reshape([1, D, D, 16]);
        new_state = new_state.clipByValue(-3., +3.);
        state.assign(new_state);
      });
    }

    function render() {
      if (!isInViewport(canvas)) {
        requestAnimationFrame(render);
        return;
      } 
      step();

      const imageData = tf.tidy(()=>{
        const rgba = state.slice([0, 0, 0, 0], [-1, -1, -1, 4]);
        const a = state.slice([0, 0, 0, 3], [-1, -1, -1, 1]);
        const img = tf.tensor(1.0).sub(a).add(rgba).mul(255);
        const rgbaBytes = new Uint8ClampedArray(img.dataSync());
        return new ImageData(rgbaBytes, w, h);
      });
      ctx.putImageData(imageData, 0, 0);

      requestAnimationFrame(render);
    }
    render();
  }
  //run();
  tf.setBackend('webgl').then(run);
}
