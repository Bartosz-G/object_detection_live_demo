const classMapping = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

let offscreenCanvas;
let ctx;
let videoWidth, videoHeight;
let latency, boundingBoxes, labels;

let fontSizeRatio = 100;
let lineWidthRatio = 1000;

let lineWidth;
let lineSlider = 0.005;
let strokeStyle = 'red';
let fontSize;
let fontSlider = 0.05;



self.onmessage = (e) => {
    const { type, buffer, canvas, width, height, lineWidthNew, strokeStyleNew, fontSizeNew } = e.data;

    if (type === 'initCanvas') {
        offscreenCanvas = canvas;
        ctx = offscreenCanvas.getContext('2d');
        self.postMessage({ message: 'canvasReady' });
        console.log('initCanvas called ', videoWidth, videoHeight)

    } else if (type === 'videoMetadata') {
        videoWidth = width;
        videoHeight = height;
        adjustCanvasSize();
        console.log('videoMetadata called ', videoWidth, videoHeight)

    } else if (type === 'resize') {
        videoWidth = width;
        videoHeight = height;
        adjustCanvasSize();
        console.log('resize called: ', videoWidth, videoHeight)

    } else if (type === 'decodePrediction') {
        const {lat, lbls, bboxes} = decodeBuffer(buffer)
        latency = lat
        labels = lbls
        boundingBoxes = bboxes
        console.log('bboxes :', bboxes);
        drawBbox();
    } else if (type === 'changeLineWidth') {
        lineSlider = lineWidthNew / lineWidthRatio;
        adjustCanvasSize();
    } else if (type === 'changeFontSize') {
        fontSlider = fontSizeNew / fontSizeRatio;
        adjustCanvasSize();
    } else if (type === 'changeColour') {
        strokeStyle = strokeStyleNew;
        adjustCanvasSize();
    }
    else {
        console.log('unknown message: ', type)
    }
};

let decodeBuffer = (buffer) => {
    if (!(buffer instanceof ArrayBuffer)) {
        throw new TypeError('Expected ArrayBuffer for the first argument.');
    }


    const view = new DataView(buffer);
    let offset = 0;

    const lat = view.getUint8(offset);
    offset += 1;

    const numLabels = view.getUint32(offset, true);

    const numBboxes = view.getUint32(offset, true) / 16;
    offset += 4;

    const lbls = [];
    for (let i = 0; i < numLabels; i++) {
        lbls.push(view.getUint8(offset));
        offset += 1;
    }

    const bboxes = [];
    for (let i = 0; i <= numBboxes - 1; i++) {
        const bbox = [];
        // postMessage({message: offset})
        for (let j = 0; j < 4; j++) {
            bbox.push(view.getFloat32(offset, true));
            offset += 4;
        }
        bboxes.push(bbox);
    }

    return {lat, lbls, bboxes}
}



let adjustCanvasSize = () => {
    if (offscreenCanvas) {
        offscreenCanvas.width = videoWidth;
        offscreenCanvas.height = videoHeight;
        fontSize = fontSlider * videoWidth;
        lineWidth = lineSlider * videoWidth;
        drawBbox();
    }
}

let drawBbox = () => {
    if (!latency) {
        // Means backend isn't processing yet
        return;
    }

        if (!boundingBoxes || !labels) {
        console.error('Bounding boxes or classes are undefined.');
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = strokeStyle;
    ctx.font = `${fontSize}px Arial`;
    ctx.fillStyle = strokeStyle;
    let offsetClassText = fontSize * 0.1

    const canvasWidth = offscreenCanvas.width;
    const canvasHeight = offscreenCanvas.height;


    ctx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);


    boundingBoxes.forEach((coords, index) => {
        // console.log('bbox.forEach() called!')
        const [x, y, w, h] =  coords
        if (lineWidth != 0) {
            ctx.strokeRect(x * canvasWidth, y * canvasHeight, w * canvasWidth, h * canvasHeight);
        }

        const classes = classMapping[labels[index]];

        if (fontSlider != 0) {
            ctx.fillText(classes, x * canvasWidth, y * canvasHeight - offsetClassText);
        }
    });

    const latencyText = `${latency}ms`
    const latencyTextSize = ctx.measureText(latencyText);
    ctx.fillText(latencyText, canvasWidth - latencyTextSize.width, latencyTextSize.actualBoundingBoxAscent + latencyTextSize.actualBoundingBoxDescent);
}