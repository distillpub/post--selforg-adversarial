function mnistDemo() {
    'use strict';

    const $ = q=>document.querySelector(q);

    const decodeFloat32 = b64 =>
        fetch("data:application/octet-binary;base64," + b64)
            .then(res => res.arrayBuffer())
            .then(buffer => tf.tensor(new Float32Array(buffer)));

    const loadImage = src => new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = src;
    });

    async function getWeights(w_b64) {
        let [w0, b0, w1, b1, w2, b2] = await Promise.all(w_b64.map(decodeFloat32));
        w0 = w0.reshape([3 * 3 * 20, 80]);
        w1 = w1.reshape([80, 80]);
        w2 = w2.reshape([80, 19]);

        return [w0, b0, w1, b1, w2, b2];
    }

    let drawadversaryCkbx = document.getElementById("drawadversary");

    async function main() {

        const colorLookup = tf.tensor([
            [128, 0, 0],
            [230, 25, 75],
            [70, 240, 240],
            [210, 245, 60],
            [250, 190, 190],
            [170, 110, 40],
            [170, 255, 195],
            [165, 163, 159],
            [0, 128, 128],
            [128, 128, 0],
            [0, 0, 0], // This is the default for digits.
        ], null, 'int32')

        const weights = await getWeights(WEIGHTS_B64);
        const adv_weights = await getWeights(ADV_WEIGHTS_B64);
        const initImg = await loadImage('digits.png');

        const [W, H, CH] = [140, 140, 20];
        const ALIVE_ALPHA = 0.1
        const state = tf.variable(tf.zeros([H, W, CH]))
        window.state = state;
        const livingCoords = [];
        const advLivingCoords = [];

        const canvas = $('canvas');
        canvas.width = W;
        canvas.height = H;
        canvas.style.width = W*4+'px'
        canvas.style.height = H*4+'px'
        const ctx = canvas.getContext('2d')
        let imageData = ctx.getImageData(0, 0, W, H);
        ctx.drawImage(initImg, 0, 0);


        const adv_canvas = new OffscreenCanvas(W, H);
        const adv_ctx = adv_canvas.getContext('2d');
        let advImageData = adv_ctx.getImageData(0, 0, W, H);


        const syncCanvas = ()=>tf.tidy(() => {
            const prevImageData = imageData;
            imageData = ctx.getImageData(0, 0, W, H);
            const advPrevImageData = advImageData;
            advImageData = adv_ctx.getImageData(0, 0, W, H);
            const buf = state.dataSync();
            livingCoords.length = 0;
            advLivingCoords.length = 0;
            for (let i=0; i<H*W; ++i) {
                const alphaOfs = i*4+3;
                const a0 = prevImageData.data[alphaOfs];
                const a = imageData.data[alphaOfs];
                const adv_a0 = advPrevImageData.data[alphaOfs];
                const adv_a = advImageData.data[alphaOfs];
                if (a>ALIVE_ALPHA*255) {
                    buf[i*CH] = a/255.0;
                    if (adv_a>ALIVE_ALPHA*255) {
                        advLivingCoords.push([Math.floor(i/W), i%W]);
                    } else {
                        livingCoords.push([Math.floor(i/W), i%W]);
                    }
                } else if (a!=a0) {
                    buf.fill(0.0, i*CH, i*CH+CH);
                    imageData.data[alphaOfs] = 0;
                }
            }
            state.assign(tf.tensor(buf, state.shape));
        });
        syncCanvas();

        const hood1d = [-1, 0, 1];
        //const hood2d = tf.tensor(hood1d.map(y=>hood1d.map(x=>[y, x])).flat(), null, 'int32');
        const hood2d = hood1d.map(y => hood1d.map(x => tf.tensor([y, x], [2], 'int32'))).flat();

        const runModel = (weights, state, idx) => {
            // there is no tf.tidy here.
            const [w0, b0, w1, b1, w2, b2] = weights;
            const hoodIdx = tf.stack(hood2d.map(ofs => idx.add(ofs)), 1); // [n, 9, 2]
            let x = tf.gatherND(state, hoodIdx).reshape([-1, w0.shape[0]]); // [n, 9*20]
            x = tf.fused.matMul({ a: x, b: w0, bias: b0, activation: 'relu' });
            x = tf.fused.matMul({ a: x, b: w1, bias: b1, activation: 'relu' });
            x = tf.fused.matMul({ a: x, b: w2, bias: b2 });
            x = x.add(tf.randomNormal(x.shape, 0.0, 0.02));
            x = x.pad([[0, 0], [1, 0]]);
            x = tf.scatterND(idx, x, state.shape);
            return x;
        }

        // add blinking capability.
        const initT = new Date().getTime() / 1000;
        const step = ()=>tf.tidy(() => {
            const old_state = state;
            if (livingCoords.length > 0) {
                tf.util.shuffle(livingCoords);
                const idxList = livingCoords.slice(0, Math.max(livingCoords.length/2, 1));
                const idx = tf.tensor(idxList, null, 'int32');
                const x = runModel(weights, old_state, idx);
                state.assign(state.add(x));
                            
                // update vis
                const label = tf.gatherND(state, idx).slice([0, 10], [-1, -1]).pad([[0, 0], [0, 1]], 0.1).argMax(-1);
                const colors = colorLookup.gather(label).dataSync();
                for (let i=0; i<idxList.length; ++i) {
                    const [y, x] = idxList[i];
                    const p = (y*W+x)*4;
                    imageData.data[p] = colors[i*3];
                    imageData.data[p+1] = colors[i*3+1];
                    imageData.data[p+2] = colors[i*3+2];
                }
            }
            if (advLivingCoords.length > 0 && (advLivingCoords.length>1 || Math.random() < 0.5)) {
                tf.util.shuffle(advLivingCoords);
                const idxList = advLivingCoords.slice(0, Math.max(advLivingCoords.length/2, 1));
                const idx = tf.tensor(idxList, null, 'int32');
                const x = runModel(adv_weights, old_state, idx);
                state.assign(state.add(x));
                            
                // update vis. Since we want blinking, this updates all of these pixels (not 50%).
                const label = tf.gatherND(state, advLivingCoords).slice([0, 10], [-1, -1]).pad([[0, 0], [0, 1]], 0.1).argMax(-1);
                const colors = colorLookup.gather(label).dataSync();
                const seconds = new Date().getTime() / 1000 - initT;
                const t = tf.tensor(seconds).sin().mul(0.5).add(0.5).dataSync();
                const onemt = 1. - t;
                for (let i=0; i<advLivingCoords.length; ++i) {
                    const [y, x] = advLivingCoords[i];
                    const p = (y*W+x)*4;
                    imageData.data[p] = colors[i*3] * onemt + (15 * t);
                    imageData.data[p+1] = colors[i*3+1] * onemt;
                    imageData.data[p+2] = colors[i*3+2] * onemt;
                }
            }

        });
        step(); // warm up


        let drawRadius = 1.0;
        const isErasing = e=>!$("#pen").checked || e.shiftKey;


        ctx.strokeStyle = "#000000";
        ctx.fillStyle = "#000000";
        adv_ctx.strokeStyle = "#000000";
        adv_ctx.fillStyle = "#000000";
        const line = (x0, y0, x1, y1, e) => {
            let r = drawRadius;
            if (isErasing(e)) {
                ctx.globalCompositeOperation = "destination-out";
                r *= 5.0;
            }
            ctx.lineWidth = r*2.0;
            ctx.beginPath();
            ctx.moveTo(x0, y0);
            ctx.lineTo(x1, y1);
            ctx.stroke();
            ctx.globalCompositeOperation = "source-over";

            if (drawadversaryCkbx.checked || isErasing(e)) {
                if (isErasing(e)) {
                    adv_ctx.globalCompositeOperation = "destination-out";
                }
                adv_ctx.lineWidth = r*2.0;
                adv_ctx.beginPath();
                adv_ctx.moveTo(x0, y0);
                adv_ctx.lineTo(x1, y1);
                adv_ctx.stroke();
                adv_ctx.globalCompositeOperation = "source-over";
            }

        }

        // make it a point!
        const pointDraw = (x, y) => {
            ctx.fillRect(x, y, 1, 1);
            ctx.globalCompositeOperation = "source-over";
            if (drawadversaryCkbx.checked) {
                // Perform surgical insertion of 1 pixel only.
                adv_ctx.fillRect(x, y, 1, 1);
                adv_ctx.globalCompositeOperation = "source-over";
            }
        }

        const circle = (x, y, e) => {
            if (drawadversaryCkbx.checked && !isErasing(e)) {
                pointDraw(x, y);
                return;
            }
            let r = drawRadius;
            if (isErasing(e)) {
                ctx.globalCompositeOperation = "destination-out";
                r *= 5.0;
            }
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fill();
            ctx.globalCompositeOperation = "source-over";
            if (drawadversaryCkbx.checked && isErasing(e)) {
                // the isErasing call is superfluous, but kept for clarity.
                adv_ctx.globalCompositeOperation = "destination-out";
                adv_ctx.beginPath();
                adv_ctx.arc(x, y, r, 0, 2 * Math.PI);
                adv_ctx.fill();
                adv_ctx.globalCompositeOperation = "source-over";
            }
        }

        function canvasToGrid(xin, yin) {
            const x = xin / canvas.clientWidth * W;
            const y = yin / canvas.clientHeight * H;
            return [x, y];    
        }
        const getClickPos = e=>{
            return canvasToGrid(e.offsetX, e.offsetY);
        }

        function getTouchPos(touch) {
            const rect = canvas.getBoundingClientRect();
            return canvasToGrid(touch.clientX - rect.left, touch.clientY - rect.top);
        }
        let lastPos = 0;

        canvas.onmousedown = e => {
            const [x, y] = getClickPos(e);
            lastPos = [x, y];
            circle(x, y, e);
            syncCanvas();
        }
        canvas.onmousemove = e => {
            const [x, y] = getClickPos(e);
            if (e.buttons == 1) {
                const [x0, y0] = lastPos;
                circle(x, y, e);
                line(x0, y0, x, y, e);
                syncCanvas();
            }
            lastPos = [x, y];
        }

        let lastTouchId = 0;
        let lastTouchList = null;

        canvas.addEventListener("touchstart", e => {
            e.preventDefault();
            for (const t of e.changedTouches) {
                const [x, y] = getTouchPos(t);
                circle(x,y, e);
            }
            syncCanvas();
            lastTouchList = e.touches;
        });

        canvas.addEventListener("touchmove", e => {
            e.preventDefault();
            for (const t of e.changedTouches) {
                const [x, y] = getTouchPos(t);
                for (const tOld of lastTouchList){
                    if (t.identifier == tOld.identifier) {
                        const [xOld, yOld] = getTouchPos(tOld);
                        line(xOld, yOld, x, y, e);
                    }
                }
                circle(x, y, e);
            }
            syncCanvas();
            lastTouchList = e.touches; 
        });

        $('#clearBtn').onclick = ()=>{
            ctx.clearRect(0, 0, W, H);
            adv_ctx.clearRect(0, 0, W, H);
            syncCanvas();
        }
        $('#resetBtn').onclick = ()=>{
            ctx.clearRect(0, 0, W, H);
            adv_ctx.clearRect(0, 0, W, H);
            syncCanvas();
            ctx.drawImage(initImg, 0, 0);
            syncCanvas();
        }
        $('#removeadvBtn').onclick = ()=>{
            adv_ctx.clearRect(0, 0, W, H);
            syncCanvas();
        }

        function render() {
            const t0 = Date.now();
            step();
            const dt = Math.max(Date.now()-t0, 1);
            const fps = Math.round(1000.0 / dt);
            $('#log').innerText = `${fps} fps`
            ctx.putImageData(imageData, 0, 0)
            requestAnimationFrame(render);
        }
        render();
    }

    tf.setBackend('wasm').then(main);
}
mnistDemo();
