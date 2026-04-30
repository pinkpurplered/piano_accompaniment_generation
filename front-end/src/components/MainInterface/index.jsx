import React, { useState, useEffect, useRef } from 'react';
import { message } from 'antd';
import { YoutubeOutlined, CustomerServiceOutlined, LoadingOutlined } from '@ant-design/icons';
import axios from 'axios';
import { myServer } from '../../utils';
import './index.css';

const YT_UPLOAD_TIMEOUT_MS = 50 * 60 * 1000;
const GENERATED_QUERY_TIMEOUT_MS = 25 * 60 * 1000;

const statusText = [
    'Preparing audio...',
    'Extracting melody & analyzing...',
    'Generating piano accompaniment (LLaMA-MIDI)...',
    'Continuing generation...',
    'Parsing MIDI output...',
    'Synthesizing MIDI...',
    'Finalizing MIDI...',
    'Mixing MP3 audio (this may take a few minutes)...'
];

export default function MainInterface() {
    const [url, setUrl] = useState('');
    const [status, setStatus] = useState('idle'); // idle, processing, done, error
    const [stage, setStage] = useState(0);
    const [errorMsg, setErrorMsg] = useState(null);
    const [chordName, setChordName] = useState(null);
    const [accName, setAccName] = useState(null);
    const [vocalChordMp3, setVocalChordMp3] = useState(null);
    const [vocalTexturedMp3, setVocalTexturedMp3] = useState(null);
    const intervalRef = useRef(null);

    useEffect(() => {
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, []);

    const onGenerate = () => {
        const u = url.trim();
        if (!u) {
            message.warn('Please enter a YouTube URL.');
            return;
        }

        setStatus('processing');
        setStage(0);
        setErrorMsg(null);
        setChordName(null);
        setAccName(null);
        setVocalChordMp3(null);
        setVocalTexturedMp3(null);

        axios.post(`${myServer}/upload_youtube`, { url: u, use_vocal_only: true }, {
            withCredentials: true,
            timeout: YT_UPLOAD_TIMEOUT_MS,
        })
        .then((res) => {
            if (res.data && res.data.status === 'ok') {
                startGeneration(res.data);
            } else {
                setStatus('error');
                setErrorMsg(res.data?.status || 'YouTube load failed');
            }
        })
        .catch((e) => {
            setStatus('error');
            setErrorMsg(e.message || 'Request failed');
        });
    };

    const startGeneration = (d) => {
        const tonic = d.suggested_tonic || 'C';
        const mode = d.suggested_mode || 'maj';
        const tempo = d.detected_tempo ? Math.max(30, Math.min(260, Math.round(Number(d.detected_tempo)))) : 120;

        const payload = {
            tonic, mode, tempo
        };

        axios.post(`${myServer}/generate`, payload, { withCredentials: true })
            .then(() => {
                intervalRef.current = setInterval(askStage, 2000);
            })
            .catch(e => {
                setStatus('error');
                setErrorMsg('Failed to start generation');
            });
    };

    const askStage = () => {
        axios.get(`${myServer}/stage_query`, { withCredentials: true })
            .then(res => {
                const data = res.data;
                if (data.status !== 'ok') return;
                
                if (data.generate_error) {
                    clearInterval(intervalRef.current);
                    setStatus('error');
                    setErrorMsg(data.generate_error);
                    return;
                }
                
                const currentStage = parseInt(data.stage || 0);
                setStage(currentStage);
                
                if (currentStage === 7) {
                    clearInterval(intervalRef.current);
                    fetchResults();
                }
            })
            .catch(() => {});
    };

    const fetchResults = () => {
        setStage(7);
        axios.get(`${myServer}/generated_query`, {
            withCredentials: true,
            timeout: GENERATED_QUERY_TIMEOUT_MS,
        })
            .then(res => {
                if (res.data.status === 'ok') {
                    setChordName(res.data.chord_midi_name);
                    setAccName(res.data.acc_midi_name);
                    setVocalChordMp3(res.data.vocal_chord_mp3);
                    setVocalTexturedMp3(res.data.vocal_textured_mp3);
                    setStatus('done');
                } else {
                    setStatus('error');
                    setErrorMsg(res.data?.status || 'Failed to fetch generated files');
                }
            })
            .catch((e) => {
                setStatus('error');
                const backendMsg = e?.response?.data?.status;
                if (backendMsg) {
                    setErrorMsg(backendMsg);
                    return;
                }
                if (e?.code === 'ECONNABORTED') {
                    setErrorMsg('Result assembly timed out while mixing audio. Please try a shorter song or try again.');
                    return;
                }
                setErrorMsg(e?.message || 'Failed to fetch generated files');
            });
    };

    return (
        <div className="app-container">
            <div className="main-content">
                <h1 className="hero-title">Piano Accompaniment Generation</h1>
                <p className="hero-subtitle">
                    Transform any YouTube song into a piano accompaniment MIDI.<br/>
                    Powered by AI, completely automatic.
                </p>

                <div className="input-wrapper">
                    <YoutubeOutlined style={{ fontSize: '24px', color: '#ff0000', marginRight: '12px' }} />
                    <input 
                        className="custom-input"
                        placeholder="Paste a YouTube link here..."
                        value={url}
                        onChange={e => setUrl(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && status !== 'processing' && onGenerate()}
                        disabled={status === 'processing'}
                    />
                    <button 
                        className="generate-btn" 
                        onClick={onGenerate}
                        disabled={status === 'processing' || !url.trim()}
                    >
                        {status === 'processing' ? 'Generating...' : 'Generate'}
                    </button>
                </div>

                {status === 'processing' && (
                    <div className="status-area">
                        <LoadingOutlined style={{ fontSize: '36px', color: '#111' }} spin />
                        <div className="status-text">{statusText[Math.min(stage, 7)]}</div>
                        <div className="status-subtext">
                            {stage === 0 ? "Isolating vocals. This takes a few minutes..." : "Composing accompaniment..."}
                        </div>
                    </div>
                )}

                {status === 'error' && (
                    <div className="error-box">
                        <div style={{ fontWeight: 600, marginBottom: '8px', fontSize: '1.1rem' }}>Generation Failed</div>
                        <div>{errorMsg}</div>
                        <button className="reset-btn" onClick={() => setStatus('idle')} style={{ marginTop: '20px' }}>Try Again</button>
                    </div>
                )}

                {status === 'done' && (
                    <div className="status-area">
                        <h2 style={{ marginBottom: '24px', fontSize: '1.5rem', fontWeight: 600 }}>Your Accompaniments</h2>
                        
                        <div className="results-grid">
                            {/* MP3 with audio player: Vocals + Block Chords */}
                            {vocalChordMp3 && (
                                <div className="result-card audio-card">
                                    <CustomerServiceOutlined className="result-icon" />
                                    <div className="result-title">Vocals + Block Chords</div>
                                    <div className="result-desc">Isolated vocals + generated grand piano</div>
                                    <audio controls className="audio-player" preload="metadata">
                                        <source src={`${myServer}/mp3/${vocalChordMp3}`} type="audio/mpeg" />
                                        Your browser does not support the audio element.
                                    </audio>
                                </div>
                            )}
                            
                            {/* MP3 with audio player: Vocals + Textured Piano */}
                            {vocalTexturedMp3 && (
                                <div className="result-card audio-card">
                                    <CustomerServiceOutlined className="result-icon" />
                                    <div className="result-title">Vocals + Textured Piano</div>
                                    <div className="result-desc">Isolated vocals + generated grand piano texture</div>
                                    <audio controls className="audio-player" preload="metadata">
                                        <source src={`${myServer}/mp3/${vocalTexturedMp3}`} type="audio/mpeg" />
                                        Your browser does not support the audio element.
                                    </audio>
                                </div>
                            )}
                            
                            {/* MIDI download: Chord Accompaniment Only */}
                            {chordName && (
                                <a href={`${myServer}/midi/${chordName}`} download="chord_accompaniment.mid" className="result-card">
                                    <CustomerServiceOutlined className="result-icon" />
                                    <div className="result-title">Block Chords MIDI</div>
                                    <div className="result-desc">Accompaniment only (no vocals)</div>
                                </a>
                            )}
                            
                            {/* MIDI download: Textured Chord Accompaniment Only */}
                            {accName && (
                                <a href={`${myServer}/midi/${accName}`} download="textured_accompaniment.mid" className="result-card">
                                    <CustomerServiceOutlined className="result-icon" />
                                    <div className="result-title">Textured Piano MIDI</div>
                                    <div className="result-desc">Accompaniment only (no vocals)</div>
                                </a>
                            )}
                        </div>
                        
                        <button className="reset-btn" onClick={() => { setStatus('idle'); setUrl(''); setVocalChordMp3(null); setVocalTexturedMp3(null); }}>
                            Generate Another Song
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}