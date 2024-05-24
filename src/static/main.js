const stunServers = [
    {
        urls: ['stun:stun.kinesisvideo.eu-west-2.amazonaws.com:443', 'stun.l.google.com']
    }
];

const ip = '13.42.36.213';
const offerEndpoint = '/offer';

let localStream;
let remoteStream;
let peerConnection;

const displayDimensions = {
    width: { ideal: 1280 },
    height: { ideal: 720 }}

const classifyDimensions =  {
    width: { exact: 640 },
    height: { exact: 640 }
  }

let websocket;
var client_id = Date.now();
let websocket_url = 'ws://' + window.location.hostname + `:80/ws/${client_id}`;




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


let decodePrediction = (pred) => {
    const buffer = pred.data;
    const dataView = new DataView(buffer);

    // Read header information
    const latency = dataView.getUint8(0);           // Read the first byte for the single uint8 value
    const lengthLabels = dataView.getUint32(1, false); // Read next 4 bytes for tensor1 length
    const lengthBboxes = dataView.getUint32(5, false); // Read next 4 bytes for tensor2 length

    // Calculate offsets
    const offsetLabels = 9; // Header is 9 bytes (1 + 4 + 4)
    const offsetBboxes = offsetLabels + lengthLabels;

    // Extract tensors
    const labelsData = new Uint8Array(buffer, offsetLabels, lengthLabels);
    const bboxesData = new Float32Array(buffer, offsetBboxes, lengthBboxes / 4); // Each float32 is 4 bytes

    // Process tensors and uint8 value (example)
    // console.log('Received latency:', latency);
    // console.log('Received labels data:', labelsData);
    // console.log('Received bounding boxes data:', bboxesData);

    // Convert Uint8Array and Float32Array to standard arrays if needed
    const labelsArray = Array.from(labelsData);
    const bboxesArray = [];
    for (let i = 0; i < bboxesData.length; i += 4) {
        bboxesArray.push(bboxesData.slice(i, i + 4));
    }

    return { latency, labelsArray, bboxesArray }

    // console.log('Labels as array:', labelsArray);
    // console.log('Bounding boxes as array of arrays:', bboxesArray);
};

let onPredictionReceived = (pred) => {
    let { latency, labelsArray, bboxesArray } = decodePrediction(pred);
    console.log('[DataChannel] latency:', latency);
}



let init = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('MediaDevices API not supported or blocked in this browser.');
        return;
    }
    try {
        localStream = await navigator.mediaDevices.getUserMedia({video: displayDimensions, audio: false});
        document.getElementById('viewer').srcObject = localStream;
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
