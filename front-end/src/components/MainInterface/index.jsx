import React, { useState, useEffect, useRef } from 'react';
import { message } from 'antd';
import { YoutubeOutlined, CustomerServiceOutlined, LoadingOutlined } from '@ant-design/icons';
import axios from 'axios';
import { myServer, myRoot } from '../../utils';
import './index.css';

const YT_UPLOAD_TIMEOUT_MS = 50 * 60 * 1000;

const statusText = [
    'Preparing audio...',
    'Extracting melody & analyzing...',
    'Constructing chord progressions...',
    'Refining progressions...',
    'Generating textures...',
    'Synthesizing MIDI...',
    'Complete!'
];

export default function MainInterface() {
    const [url, setUrl] = useState('');
    const [status, setStatus] = useState('idle'); // idle, processing, done, error
    const [stage, setStage] = useState(0);
    const [errorMsg, setErrorMsg] = useState(null);
    const [chordName, setChordName] = useState(null);
    const [accName, setAccName] = useState(null);
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
        const phrases = (d.auto_phrases && d.auto_phrases.length) ? d.auto_phrases : [{ phrase_name: 'A', phrase_length: 8 }];
        const tonic = d.suggested_tonic || 'C';
        const mode = d.suggested_mode || 'maj';
        const tempo = d.detected_tempo ? Math.max(30, Math.min(260, Math.round(Number(d.detected_tempo)))) : 120;

        const payload = {
            tonic, mode, tempo, meter: '4/4', phrases,
            chord_style: 'pop_standard', rhythm_density: 2, voice_number: 2,
            enable_texture_style: true, enable_chord_style: true, use_vocal_only: true
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
        axios.get(`${myServer}/generated_query`, { withCredentials: true })
            .then(res => {
                if (res.data.status === 'ok') {
                    setChordName(res.data.chord_midi_name);
                    setAccName(res.data.acc_midi_name);
                    setStatus('done');
                } else {
                    setStatus('error');
                    setErrorMsg('Failed to fetch generated files');
                }
            })
            .catch(() => {
                setStatus('error');
                setErrorMsg('Failed to fetch generated files');
            });
    };

    return (
        <div className="app-container">
            <div className="main-content">
                <h1 className="hero-title">Piano Accompaniment</h1>
                <p className="hero-subtitle">
                    Transform any YouTube song into a beautiful piano accompaniment MIDI.<br/>
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
                        <div className="status-text">{statusText[Math.min(stage, 6)]}</div>
                        <div className="status-subtext">
                            {stage === 0 ? "Downloading and isolating vocals. This takes a few minutes..." : "Composing accompaniment..."}
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
                        <div className="results-grid">
                            <a href={`${myRoot}/midi/${chordName}`} download="chord.mid" className="result-card">
                                <CustomerServiceOutlined className="result-icon" />
                                <div className="result-title">Chords</div>
                                <div className="result-desc">Melody + Block Chords</div>
                            </a>
                            <a href={`${myRoot}/midi/${accName}`} download="accompaniment.mid" className="result-card">
                                <CustomerServiceOutlined className="result-icon" />
                                <div className="result-title">Accompaniment</div>
                                <div className="result-desc">Melody + Textured Piano</div>
                            </a>
                        </div>
                        <button className="reset-btn" onClick={() => { setStatus('idle'); setUrl(''); }}>
                            Generate Another Song
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}