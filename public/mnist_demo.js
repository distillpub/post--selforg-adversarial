import {isInViewport} from "./util.js"

export function mnistDemo(divId, canvasId) {
    const root = document.getElementById(divId);
    const $ = q=>root.querySelector(q);
    const $$ = q => root.querySelectorAll(q);
    const $$$ = q => document.documentElement.querySelectorAll(q);
    const mnistCanvas = document.createElement('canvas');
    const mnistCtx = mnistCanvas.getContext('2d');

    let currDig = 0;
    let currSample = 0;
    let uiDigitSamples = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    let paused = false;
    let tool = 'pencil';

    let drawRadius = 1.0;

    const D = 28 * 2;
    const [W, H, CH] = [D, D, 20];

    const colors = [
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
    ];


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

    async function main() {

        const canvas = document.getElementById(canvasId);
        canvas.width = W;
        canvas.height = H;
        canvas.style.width = W*10+'px';
        canvas.style.height = H*10+'px';
        const ctx = canvas.getContext('2d');


        const colorLookup = tf.tensor(colors, null, 'int32');

        const weights = await getWeights(WEIGHTS_B64);
        const adv_weights = await getWeights(ADV_WEIGHTS_B64);
        //const initImg = await loadImage('digits.png');


        const ALIVE_ALPHA = 0.1
        const state = tf.variable(tf.zeros([H, W, CH]))
        window.state = state;
        const livingCoords = [];
        let advLivingCoords = [];

        let imageData = ctx.getImageData(0, 0, W, H);

        const syncCanvas = ()=>tf.tidy(() => {
            const prevImageData = imageData;
            imageData = ctx.getImageData(0, 0, W, H);
            const buf = state.dataSync();
            livingCoords.length = 0;
            const advCoordsMap = new Map(advLivingCoords.map(p=>[p.toString(), p]));
            advLivingCoords = Array.from(advCoordsMap.values()); // remove duplicates
            for (let i=0; i<H*W; ++i) {
                const alphaOfs = i*4+3;
                const a0 = prevImageData.data[alphaOfs];
                const a = imageData.data[alphaOfs];
                if (a>ALIVE_ALPHA*255) {
                    buf[i*CH] = a/255.0;
                    const yx = [Math.floor(i/W), i%W];
                    if (!advCoordsMap.has(yx.toString())) {
                        livingCoords.push(yx);
                    }
                } else if (a!=a0) {
                    buf.fill(0.0, i*CH, i*CH+CH);
                    imageData.data[alphaOfs] = 0;
                }
            }
            state.assign(tf.tensor(buf, state.shape));
            advLivingCoords = advLivingCoords.filter(yx=>{
                const [y, x] = yx;
                return imageData.data[(y*W+x)*4+3]>ALIVE_ALPHA*255;
            });
            console.log(advLivingCoords);
        });
        syncCanvas();


        // UI RELATED FUNCTIONS
        async function loadMnistSamples() {

            async function toBmp(url) {
                return new Promise((resolve,reject) => {
                    let img = document.createElement('img');
                    img.addEventListener('load', function() {
                        resolve(this);
                    });
                    img.src = url;
                });
            }
            const mnistBmp = await toBmp("mnist.png");  //await (await fetch("mnist.png")).blob());
            mnistCanvas.width = mnistBmp.width;
            mnistCanvas.height = mnistBmp.height;
            mnistCtx.drawImage(mnistBmp,0,0);  
        }

        function rgb(values) {
            return 'rgb(' + values.join(', ') + ')';
        }

        function getDigit(digit, sample) {
            const x = sample * 28;
            const y = digit * 28;
            return mnistCtx.getImageData(x, y, 28, 28);
        };

        function reset() {
            ctx.clearRect(0, 0, W, H);
            advLivingCoords.length = [];
            const digit = getDigit(currDig, currSample);
            const toDraw = convertDigitToDraw(digit);
            const padding = (D - 28)/2.0;
            ctx.putImageData(toDraw, padding, padding);
            syncCanvas();
        }

        function convertDigitToDraw(digit) {
            let toDraw = new ImageData(28, 28);
            console.log(toDraw.data[0], toDraw.data[1], toDraw.data[2], toDraw.data[3])
            for (let i = 0; i < 28 * 28; i++) {
                // we expect RGBA, so 4 values to process.
                const p = i*4;
                if (digit.data[p] > 25) {
                    toDraw.data[p+3] = digit.data[p];
                }
            }
            return toDraw;
        }

        function switcheroo() {
            // reset the canvas too.
            ctx.clearRect(0, 0, W, H);
            advLivingCoords.length = 0;
            // get the next digit.
            const digit = getDigit(currDig, currSample);
            const toDraw = convertDigitToDraw(digit);
            const padding = (D - 28)/2.0;
            ctx.putImageData(toDraw, padding, padding);
            syncCanvas();
        }

        //hacky way to uncolor last thing.
        async function initUI() {
            await loadMnistSamples();
            for (let i = 0; i < 10; i++) {
              const dcv = document.createElement('canvas');
              dcv.id = "digit-" + i;
              dcv.width = 28;
              dcv.height = 28;
              const dctx = dcv.getContext('2d');
              if (i != 0) {
                dctx.putImageData(getDigit(i, 0), 0, 0)
              } else {
                dctx.putImageData(getDigit(i, 1), 0, 0)
                uiDigitSamples[i] += 1;
              }
              dctx.globalCompositeOperation='difference';
              dctx.fillStyle = 'white';
              dctx.fillRect(0, 0, 28, 28);
              dctx.globalCompositeOperation = "screen";
              dctx.fillStyle = rgb(colors[i]);
              dctx.fillRect(0, 0, 28, 28);
              dcv.onclick = () => {
                // update the digit to show
                currDig = i;
                currSample = uiDigitSamples[i];
                switcheroo();
                // paint our legend with next digit
                uiDigitSamples[i] = (uiDigitSamples[i] + 1) % 20;
                dctx.putImageData(getDigit(i, uiDigitSamples[i]), 0, 0)
                dctx.globalCompositeOperation='difference';
                dctx.fillStyle = 'white';
                dctx.fillRect(0, 0, 28, 28);
                dctx.globalCompositeOperation = "screen";
                dctx.fillStyle = rgb(colors[i]);
                dctx.fillRect(0, 0, 28, 28);
                console.log("clicked" + i);
              }
              $('#pattern-selector').appendChild(dcv);
            }
            console.log("loaded");
            $('#reset').onclick = reset;
            $('#play-pause').onclick = () => {
              paused = !paused;
              updateUI();
            };
            $$('.radiotool div').forEach(e=>{ 
                e.onclick = ()=>{
                    tool = e.id;
                    updateUI();
                };
            });

            $('#bin').onclick = () => {
                ctx.clearRect(0, 0, W, H);
                advLivingCoords.length = 0;
                syncCanvas();
            };

            $('#brushSlider').oninput = (e) => {
              drawRadius = parseFloat(e.target.value)/2.0;
              updateUI();
            };

            $('#hueSlider').oninput = (e) => {
                let hue = parseFloat(e.target.value);
                $('#hueValue').innerText = hue;
                console.log($$$('.color_heavy, filter'));
                $$$('.color_heavy, figure').forEach(e => {e.style.filter = "hue-rotate(" + hue + "deg)"});
            };
            console.log("loaded");
            $('#speed').onchange = updateUI;
            $('#speed').oninput = updateUI;
            updateUI();
        };
        function updateUI() {
            $('#play').style.display = paused ? "inline" : "none";
            $('#pause').style.display = !paused ? "inline" : "none";
            const speed = parseInt($('#speed').value);
            $('#speedLabel').innerHTML = ['1/60 x', '1/10 x', '1/2 x', '1 x', '2 x', '4 x', '<b>max</b>'][speed + 3];
            $('#radius').innerText = ( (tool=='eraser') ? drawRadius * 5.0 : drawRadius);
        };


        await initUI();

        // initialize state with a digit.
        reset();

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
                const t = Math.sin(seconds*5.0)*0.5+0.5;
                for (let i=0; i<advLivingCoords.length; ++i) {
                    const [y, x] = advLivingCoords[i];
                    const p = (y*W+x)*4;
                    imageData.data[p] = colors[i*3] * t + 255 * (1.0-t);
                    imageData.data[p+1] = colors[i*3+1] * t;
                    imageData.data[p+2] = colors[i*3+2] * t;
                }
            }

        });
        step(); // warm up


        const isErasing = e=> tool=='eraser' || e.shiftKey;
        const isAdversarial = ()=>tool=='adversary';


        ctx.strokeStyle = "#000000";
        ctx.fillStyle = "#000000";
        const line = (x0, y0, x1, y1, e) => {
            if (isAdversarial())
                return;
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
        }

        const circle = (x, y, e) => {
            if (isAdversarial()) {
                advLivingCoords.push([Math.floor(y), Math.floor(x)]);
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

        $('#adversary-remove').onclick = ()=>{
            advLivingCoords.length = 0;
            syncCanvas();
        }

        function render() {
            if (!paused && isInViewport(canvas)) {
                const t0 = Date.now();
                step();
                const dt = Math.max(Date.now()-t0, 1);
                const fps = Math.round(1000.0 / dt);
                $('#ips').innerText = `${fps}`
                ctx.putImageData(imageData, 0, 0)
            }
            requestAnimationFrame(render);
        }
        requestAnimationFrame(render);
    }

    tf.setBackend('wasm').then(main);
}
