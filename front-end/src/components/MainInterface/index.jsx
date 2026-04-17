import React, { Component } from 'react'
import { Typography, Divider, Input, Button, message, Spin, Card } from 'antd';
import { LoadingOutlined, CheckCircleOutlined } from '@ant-design/icons';
import axios from 'axios';
import { myServer, server, myRoot } from '../../utils';
import Icon from '../Icon';
import './index.css'

const { Title } = Typography;
const { Meta } = Card;

const YT_UPLOAD_TIMEOUT_MS = 50 * 60 * 1000;

const statusText = [
    'Preparing...',
    'Loading melodies, initializing melody meta...',
    'Analyzing melodies, constructing progressions...',
    'Loading library, refining progressions according to styles...',
    'Generating textures...',
    'Synthesizing...',
    'Complete!',
]

export default class MainInterface extends Component {

    state = {
        url: '',
        status: 'idle', // idle, uploading, generating, done, error
        generatingStage: 0,
        generateError: null,
        generatedChordName: null,
        generatedAccName: null,
    }

    askStageInterval = null;

    componentWillUnmount() {
        if (this.askStageInterval) {
            window.clearInterval(this.askStageInterval);
        }
    }

    onGenerate = () => {
        const u = (this.state.url || '').trim();
        if (!u) {
            message.warn('Enter a YouTube or YouTube Music URL.');
            return;
        }

        this.setState({ 
            status: 'uploading', 
            generateError: null,
            generatedChordName: null,
            generatedAccName: null,
            generatingStage: 0
        });

        // Step 1: Upload YouTube
        axios.post(`${myServer}/upload_youtube`, { url: u, use_vocal_only: true }, {
            withCredentials: true,
            timeout: YT_UPLOAD_TIMEOUT_MS,
        })
        .then((res) => {
            if (res.data && res.data.status === 'ok') {
                const d = res.data;
                this.startGeneration(d);
            } else {
                this.setState({ status: 'error', generateError: res.data?.status || 'YouTube load failed' });
            }
        })
        .catch((e) => {
            this.setState({ status: 'error', generateError: e.message || 'Request failed' });
        });
    }

    startGeneration = (d) => {
        this.setState({ status: 'generating' });

        const phrases = (d.auto_phrases && d.auto_phrases.length) ? d.auto_phrases : [{ phrase_name: 'A', phrase_length: 8 }];
        const tonic = d.suggested_tonic || 'C';
        const mode = d.suggested_mode || 'maj';
        const tempo = d.detected_tempo ? Math.max(30, Math.min(260, Math.round(Number(d.detected_tempo)))) : 120;

        const payload = {
            tonic,
            mode,
            tempo,
            meter: '4/4',
            phrases,
            chord_style: 'pop_standard',
            rhythm_density: 2,
            voice_number: 2,
            enable_texture_style: true,
            enable_chord_style: true,
            use_vocal_only: true
        };

        server(`/generate`, this, null, 'post', payload);
        this.askStageInterval = setInterval(this.askStage, 2000);
    }

    askStage = () => {
        server(`/stage_query`, this, null, 'get', null, (res) => {
            if (res.status !== 'ok') return;
            
            if (res.generate_error) {
                window.clearInterval(this.askStageInterval);
                this.setState({ status: 'error', generateError: res.generate_error });
                return;
            }
            
            const stage = parseInt(res.stage);
            this.setState({ generatingStage: stage });
            
            if (stage === 7) {
                window.clearInterval(this.askStageInterval);
                server(`/generated_query`, this, null, 'get', null, (genRes) => {
                    if (genRes.status === 'ok') {
                        this.setState({ 
                            status: 'done',
                            generatedChordName: genRes.chord_midi_name, 
                            generatedAccName: genRes.acc_midi_name 
                        });
                    } else {
                        this.setState({ status: 'error', generateError: 'Failed to fetch generated files' });
                    }
                });
            }
        });
    }

    render() {
        const { status, generatingStage, generateError, generatedChordName, generatedAccName } = this.state;

        return (
            <div>
                <div className='head'>
                    <a href='/accomontage2-online'>
                        <Title level={1} style={{float:'left', marginTop:'0px', fontSize:'55px'}} className='title'>
                            <span style={{color:'#003b76'}}>A</span>cco<span style={{color:'#003b76'}}>M</span>ontage<span style={{color:'#003b76'}}>2</span>
                        </Title>
                    </a>
                </div>
                <div style={{ clear: 'both' }}></div>
                <Divider style={{marginBottom:'50px', backgroundColor:'#003b76'}} />
                
                <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
                    <Title level={3}>Generate Accompaniment from YouTube</Title>
                    <p style={{ color: '#666', marginBottom: '30px' }}>
                        Paste a YouTube link below. We will automatically extract the melody, analyze the key and tempo, and generate the chords and accompaniment.
                    </p>

                    <Input.Search
                        size="large"
                        placeholder="https://www.youtube.com/watch?v=..."
                        enterButton="Generate"
                        value={this.state.url}
                        onChange={e => this.setState({ url: e.target.value })}
                        onSearch={this.onGenerate}
                        loading={status === 'uploading' || status === 'generating'}
                        disabled={status === 'uploading' || status === 'generating'}
                    />

                    {status === 'uploading' && (
                        <div style={{ marginTop: '40px' }}>
                            <Spin size="large" />
                            <p style={{ marginTop: '20px', fontSize: '16px' }}>Downloading and isolating vocals (this may take a few minutes)...</p>
                        </div>
                    )}

                    {status === 'generating' && (
                        <div style={{ marginTop: '40px', textAlign: 'left', maxWidth: '400px', margin: '40px auto 0' }}>
                            {statusText.map((item, idx) => {
                                if (idx < generatingStage || (idx === generatingStage && generatingStage === 6)) {
                                    return <p key={idx}><CheckCircleOutlined style={{ marginRight: '6px', color: '#52c41a' }} />{item}</p>
                                } else if (idx === generatingStage) {
                                    return <p key={idx}><LoadingOutlined style={{ marginRight: '6px', color: '#1890ff' }} />{item}</p>
                                } else {
                                    return <p key={idx} style={{ color: '#ccc' }}>{item}</p>
                                }
                            })}
                        </div>
                    )}

                    {status === 'error' && (
                        <div style={{ marginTop: '40px' }}>
                            <Title level={4} style={{ color: '#cf1322' }}>Generation Failed</Title>
                            <p style={{ color: '#cf1322' }}>{generateError}</p>
                            <Button onClick={() => this.setState({ status: 'idle' })}>Try Again</Button>
                        </div>
                    )}

                    {status === 'done' && (
                        <div style={{ marginTop: '50px' }}>
                            <Title level={3} style={{ color: '#52c41a' }}><CheckCircleOutlined /> Generation Complete!</Title>
                            <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: '30px' }}>
                                <a download='chord.mid' href={`${myRoot}/midi/${generatedChordName}`}>
                                    <Card hoverable style={{ width: '300px' }}>
                                        <Meta
                                            avatar={<Icon which='midi' />}
                                            title="Download Chords"
                                            description="MIDI with melody and chords"
                                        />
                                    </Card>
                                </a>
                                <a download='accompaniment.mid' href={`${myRoot}/midi/${generatedAccName}`}>
                                    <Card hoverable style={{ width: '300px' }}>
                                        <Meta
                                            avatar={<Icon which='midi' />}
                                            title="Download Accompaniment"
                                            description="MIDI with melody and textured accompaniment"
                                        />
                                    </Card>
                                </a>
                            </div>
                            <Button style={{ marginTop: '40px' }} onClick={() => this.setState({ status: 'idle', url: '' })}>Generate Another</Button>
                        </div>
                    )}
                </div>
            </div>
        )
    }
}
