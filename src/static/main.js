// WebRTC variables
// const stunServers = [
//     {
//         urls: ['stun:stun.kinesisvideo.eu-west-2.amazonaws.com:443', 'stun.l.google.com']
//     }
// ];

// const ip = '13.42.36.213';

let localStream;
let peerConnection;
const worker = new Worker('/static/worker.js');
const video = document.getElementById('viewer');
const canvas = document.getElementById('overlay');

const displayDimensions = {
    width: { ideal: 1280 },
    height: { ideal: 720 }
}



let websocket;
var client_id = Date.now();
let websocket_url = 'ws://' + window.location.hostname + `:80/ws/${client_id}`;




// Control flow variables
const topClassesSlider = document.getElementById('topClasses');
const topClassesValue = document.getElementById('topClassesValue');
const lineWidthSlider = document.getElementById('lineWidth');
const lineWidthValue = document.getElementById('lineWidthValue');
const fontSizeSlider = document.getElementById('fontSize');
const fontSizeValue = document.getElementById('fontSizeValue');
const colorDropdown = document.getElementById('color');

const defaultTopClasses = 100;
const defaultLineWidth = 5;
const defaultFontSize = 5;
const defaultColor = 'red';


topClassesSlider.value = defaultTopClasses;
topClassesValue.value = defaultTopClasses;
lineWidthSlider.value = defaultLineWidth;
lineWidthValue.value = defaultLineWidth;
fontSizeSlider.value = defaultFontSize;
fontSizeValue.value = defaultFontSize;
colorDropdown.value = defaultColor;

// Control flow code
let validateInput = (inputElement, sliderElement, minValue, maxValue) => {
    inputElement.addEventListener('input', () => {
        let value = inputElement.value;

        // Check if the value is a number
        if (isNaN(value)) {
            inputElement.value = sliderElement.value;
            return;
        }

        value = parseFloat(value);

        // Validate and constrain the value within the min and max range
        if (value < minValue) {
            value = minValue;
        } else if (value > maxValue) {
            value = maxValue;
        }

        // Update the input and slider values
        inputElement.value = value;
        sliderElement.value = value;
    });
}


topClassesSlider.addEventListener('input', (e) => {
    const topClasses = e.target.value;
    topClassesValue.value = topClasses;
    //TODO: Implement sending to the backend
});

lineWidthSlider.addEventListener('input', (e) => {
    const lineWidth = e.target.value;
    lineWidthValue.value = lineWidth;
    worker.postMessage({ type: 'changeLineWidth', lineWidthNew: lineWidth });
});

fontSizeSlider.addEventListener('input', (e) => {
    const fontSize = e.target.value;
    fontSizeValue.value = fontSize;
    worker.postMessage({ type: 'changeFontSize', fontSizeNew: fontSize});
});

// Event listeners for number inputs
topClassesValue.addEventListener('input', (e) => {
    const topClasses = e.target.value;
    topClassesSlider.value = topClasses;
    //TODO: Implement sending to the backend
});

lineWidthValue.addEventListener('input', (e) => {
    const lineWidth = e.target.value;
    lineWidthSlider.value = lineWidth;
    worker.postMessage({ type: 'changeLineWidth', lineWidthNew: lineWidth });
});

fontSizeValue.addEventListener('input', (e) => {
    const fontSize = e.target.value;
    fontSizeSlider.value = fontSize;
    worker.postMessage({ type: 'changeFontSize', fontSizeNew: fontSize});
});

validateInput(topClassesValue, topClassesSlider, 5, 300);
validateInput(lineWidthValue, lineWidthSlider, 0, 10);
validateInput(fontSizeValue, fontSizeSlider, 0, 10);

colorDropdown.addEventListener('change', (e) => {
    const colour = e.target.value;
    worker.postMessage({ type: 'changeColour', strokeStyleNew: colour});
});

worker.onmessage = (e) => {
    const { message } = e.data;
    console.log('[worker]: ', message)
};

topClassesSlider.addEventListener('input', (e) => {
    const topClasses = e.target.value;
    console.log('Top Classes:', topClasses);
    //TODO: Implement sending over to the backend through a websocket and adjusting the topk for preprocessor
});

lineWidthSlider.addEventListener('input', (e) => {
    const lineWidth = e.target.value;
});

fontSizeSlider.addEventListener('input', (e) => {
    const fontSize = e.target.value;
});

colorDropdown.addEventListener('change', (e) => {
    const color = e.target.value;
});





// Connection handlers
let sendMessage = async (message) => {
    if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(message);
    } else {
        console.error('WebSocket is not open. ReadyState: ', websocket.readyState);
    }
}


let createOffer = async () => {
    let offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer);


    let message = JSON.stringify({'type': 'offer', 'offer': offer})

    console.log("Local Offer set:", message);

    return message
}


let createAnswer = async () => {
    let answer = await peerConnection.createAnswer()
    await peerConnection.setLocalDescription(answer);

    sdp = {'sdp': peer_connection.localDescription};
    sendMessage(JSON.stringify(sdp));
}


let handleIncomingSDP = async ( sdp ) => {
    peerConnection.setRemoteDescription(sdp).then(() => {
        console.log('Successfully set Remote description!');
        if (sdp.type != 'offer') {
            return;
        } else {
            peerConnection.setLocalDescription(sdp)
                .then(createAnswer)
                .catch((e) => {console.error('Error while creating answer: ' + e)});
        }
    }).catch((e) => {console.error('Error while setting Remote Description: ' + e)});
}

let handleIncomingICE = async ( ice ) => {
    var candidate = new RTCIceCandidate(ice);
    peerConnection.addIceCandidate(candidate).catch((e) => {console.error('Error while handling incoming ICE: ' + e)});
}

let onIceCandidate = ( event ) => {
    if (event.candidate == null) {
        console.log("ICE Candidate was null, done with gethering ICE candidates");
        return;
    };

    sendMessage(JSON.stringify({'ice': event.candidate}));
}




let onWebsocketMessage = async (event) => {
    console.log('Received a message from websocket of type: ' + event.data)

    if (event.data === 'PONG') {
        console.log(event.data)
        return;
    }

    if (event.data === 'waitingForOffer') {
        let Offer = await createOffer();
        sendMessage(Offer);
        return;
    }

    try {
        msg = JSON.parse(event.data);
    } catch (e) {
        console.error('Failed to parse a message from the websocket' + event.data);
    }

    if (msg.sdp != null) {
        handleIncomingSDP(msg.sdp);
    } else if (msg.ice != null) {
        console.log('Received new ICE candidate: ', msg.ice)
        handleIncomingICE(msg.ice);
    } else {
        console.error('Unknown message type')
    }


}

let onWebsocketError = () => {} //TODO: Impl Websocket error handling

let onWebsocketClose = () => {} //TODO: Impl Socket Close handling

let websocketConnect = () => {} //TODO: Refactor later


let onPredictionReceived = (pred) => {
    const buff = pred.data;
    worker.postMessage({ type: 'decodePrediction', buffer: buff});
}



let init = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('MediaDevices API not supported or blocked in this browser.');
        return;
    }
    try {
        const offscreen = canvas.transferControlToOffscreen();
        worker.postMessage({ type: 'initCanvas', canvas: offscreen }, [offscreen]);

        window.addEventListener('resize', () => {
            worker.postMessage({ type: 'resize', width: video.clientWidth, height: video.clientHeight });
        });

        video.addEventListener('loadeddata', () => {
            worker.postMessage({ type: 'resize', width: video.clientWidth, height: video.clientHeight });
        });


        localStream = await navigator.mediaDevices.getUserMedia({video: displayDimensions, audio: false});
        video.srcObject = localStream;
    } catch (error) {
        console.error('Error accessing media devices.', error);
    }

    peerConnection = new RTCPeerConnection( stunServers );


    localStream.getVideoTracks().forEach((track) => {
            track.applyConstraints(classifyDimensions)
            peerConnection.addTrack(track, localStream)
        });

    peerConnection.onicecandidate = onIceCandidate;

    // ---------- Data Channel ----------
    dataChannel = peerConnection.createDataChannel("DataChannel");

    dataChannel.onopen = () => {
        console.log("Data channel is open");
        dataChannel.send("Hello from the client!");
    };

    dataChannel.onmessage = (event) => {
        onPredictionReceived(event);
    };

    dataChannel.onclose = () => {
        console.log("Data channel is closed");
    };

    dataChannel.onerror = (error) => {
        console.error("Data channel error:", error);
    };

    peerConnection.ondatachannel = (event) => {
        const receiveChannel = event.channel;
        receiveChannel.onmessage = (event) => {
            console.log("Received message from server:", event.data);
        };
    };

    // ------- Monitoring ICE negotiations ---------
    peerConnection.onicecandidateerror = e => {
        console.log("ERROR: ICE candidate Error: ", e)
    }

    peerConnection.oniceconnectionstatechange = () => {
            console.log("ICE connection state change:", peerConnection.iceConnectionState);
    }

    peerConnection.onicegatheringstatechange = () => {
        console.log("ICE gathering state change:", peerConnection.iceGatheringState);
    };
    // -----------


    try {
        websocket = new WebSocket(websocket_url)
        websocket.onmessage = onWebsocketMessage;
        websocket.onopen = (e) => {
            sendMessage("PING");
        };
    } catch (error) {
        console.error('Error accessing the websocket', error)
    }

};






init();
