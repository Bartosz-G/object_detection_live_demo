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
    const buffer = pred.data
    const view = new DataView(buffer)

    let offset = 0;

    // Read the header values
    const latency = view.getUint8(offset);
    offset += 1;

    const labelsLength = view.getUint32(offset, true); // true for little-endian
    offset += 4;

    const bboxesLength  = view.getUint32(offset, true); // true for little-endian
    offset += 4;

    const labelsData = [];
    for (let i = 0; i < labelsLength; i++) {
        labelsData.push(view.getUint8(offset));
        offset += 1;
    }

    const bboxesData = [];
    for (let i = 0; i < bboxesLength; i += 4) {
        const innerArray = [];
        for (let j = 0; j < 4; j++) {
            innerArray.push(view.getFloat32(offset, true)); // true for little-endian
            offset += 4;
        }
        bboxesData.push(innerArray);
    }
    
    return { latency, labelsData, bboxesData };

};

let onPredictionReceived = (pred) => {
    let { latency, labelsArray, bboxesArray } = decodePrediction(pred);
    console.log('[DataChannel] latency:', latency);
    console.log('[DataChannel] labels:', labelsArray)
    console.log(('[DataChannel] bboxes:', bboxesArray))
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
