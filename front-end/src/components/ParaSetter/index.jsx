import React, { Component } from 'react'
import {
    Form,
    Select,
    Button,
    Divider,
    Space,
    Slider,
    Checkbox,
    Input,
    InputNumber,
    message,
} from 'antd';
import { PlusOutlined, MinusCircleOutlined } from '@ant-design/icons';
import axios from 'axios';
import { myServer, server, meter, tonic, style, mode } from '../../utils';
import PubSub from 'pubsub-js';
const { Option } = Select;

const formItemLayout = {
    labelCol: { span: 6 },
    wrapperCol: { span: 14 },
};

/** Demucs + Basic Pitch on CPU can exceed CRA’s default proxy timeout; keep in sync with setupProxy.js */
const YT_UPLOAD_TIMEOUT_MS = 50 * 60 * 1000;

export default class ParaSetter extends Component {

    formRef = React.createRef();

    state = {
        chordStyleControl: true,
        textureStyleControl: true,
        values: null,
        melodyLoaded: false,
        youtubeLoading: false,
    }

    onFinish = (values) => {
        if (!this.state.melodyLoaded) {
            message.warn('Load the full song from YouTube first (use the Load melody button).');
            return;
        }
        const fallbackPhrases = [{ phrase_name: 'A', phrase_length: 8 }];
        let phrases = values.phrases;
        if (!Array.isArray(phrases) || phrases.length === 0) {
            phrases = fallbackPhrases;
        } else {
            phrases = phrases.filter((p) => p && p.phrase_name && p.phrase_length != null);
            if (phrases.length === 0) {
                phrases = fallbackPhrases;
            }
        }
        const payload = { ...values, phrases };
        server(`/generate`, this, null, 'post', payload);
        PubSub.publish('submit', payload);
    }

    onYoutubeLoad = (url) => {
        const u = (url || '').trim();
        if (!u) {
            message.warn('Enter a YouTube or YouTube Music URL.');
            return;
        }
        const form = this.formRef.current;
        const useVocalOnly = !!(
            form && typeof form.getFieldValue === 'function'
                ? form.getFieldValue('use_vocal_only')
                : true
        );
        this.setState({ youtubeLoading: true });
        message.info(
            'This step can take 20–60+ minutes on CPU (Demucs separation, then Basic Pitch). Watch the Flask terminal for lines starting with youtube_melody:. To go faster: set ACCOMONTAGE_YOUTUBE_NO_DEMUCS=1 and/or ACCOMONTAGE_BP_MAX_SECONDS=300 on the server, then restart Flask.',
            14,
        );
        const stopSpinning = message.loading({
            content: 'Loading melody (server is working—see Flask logs)…',
            duration: 0,
            key: 'yt_melody_loading',
        });
        axios
            .post(`${myServer}/upload_youtube`, { url: u, use_vocal_only: useVocalOnly }, {
                withCredentials: true,
                timeout: YT_UPLOAD_TIMEOUT_MS,
            })
            .then((res) => {
                const st = res.data && res.data.status;
                if (st === 'ok') {
                    const d = res.data || {};
                    const src = d.melody_source;
                    const sourceMsg =
                        src === 'vocal_stem'
                            ? 'Mode used: vocal stem (Demucs + Basic Pitch).'
                            : src === 'full_mix_user_choice'
                                ? 'Mode used: full mix (you turned vocal-only off).'
                                : src === 'full_mix_no_demucs'
                                    ? 'Mode used: full mix because Demucs is not installed.'
                                    : src === 'full_mix_fallback'
                                        ? 'Mode used: full mix fallback because vocal separation failed.'
                                        : 'Mode used: full mix.';
                    message.success(
                        `Melody loaded. ${sourceMsg} Phrases + tonic/mode were auto-filled (edit if needed). Meter is currently fixed to 4/4 for generation.`
                    );
                    this.setState({ melodyLoaded: true });
                    const applyPatch = () => {
                        const inst = this.formRef.current;
                        if (!inst || typeof inst.setFieldsValue !== 'function') {
                            return;
                        }
                        const patch = {};
                        const phrases =
                            d.auto_phrases && Array.isArray(d.auto_phrases) && d.auto_phrases.length
                                ? d.auto_phrases
                                : [{ phrase_name: 'A', phrase_length: 8 }];
                        patch.phrases = phrases;
                        if (d.suggested_tonic && tonic.includes(d.suggested_tonic)) {
                            patch.tonic = d.suggested_tonic;
                        }
                        if (d.suggested_mode === 'maj' || d.suggested_mode === 'min') {
                            patch.mode = d.suggested_mode;
                        }
                        if (d.detected_tempo) {
                            const t = Number(d.detected_tempo);
                            if (!Number.isNaN(t)) {
                                patch.tempo = Math.max(30, Math.min(260, Math.round(t)));
                            }
                        }
                        inst.setFieldsValue(patch);
                    };
                    applyPatch();
                    setTimeout(applyPatch, 0);
                    const inst = this.formRef.current;
                    if (inst && typeof inst.setFieldsValue === 'function') {
                        if (d.detected_tempo) {
                            const beatNote = d.beat_tracked
                                ? ' Tempo blends the MIDI map with beat tracking on the instrumental stem.'
                                : '';
                            message.info(
                                `Approximate tempo (for phrase guidance): ${d.detected_tempo} BPM.${beatNote}`,
                                5,
                            );
                        }
                        if (d.suggested_tonic && tonic.includes(d.suggested_tonic)) {
                            const md = d.suggested_mode === 'min' ? 'minor' : 'major';
                            message.info(
                                `Tonic / Mode suggestion: ${d.suggested_tonic} ${md} (from backing chroma and/or the melody). Adjust if wrong.`,
                                6,
                            );
                        }
                    }
                } else {
                    message.error(typeof st === 'string' ? st : 'YouTube load failed');
                    this.setState({ melodyLoaded: false });
                }
            })
            .catch((e) => {
                if (e.code === 'ECONNABORTED' || String(e.message || '').toLowerCase().includes('timeout')) {
                    message.error(
                        'Timed out waiting for the server (50 min). Try a shorter track, or on the server set ACCOMONTAGE_YOUTUBE_NO_DEMUCS=1 and/or ACCOMONTAGE_BP_MAX_SECONDS=300, restart Flask, and try again.',
                        10,
                    );
                } else {
                    const st = e.response && e.response.data && e.response.data.status;
                    message.error(st || e.message || 'Request failed');
                }
                this.setState({ melodyLoaded: false });
            })
            .finally(() => {
                if (typeof stopSpinning === 'function') {
                    stopSpinning();
                } else {
                    message.destroy('yt_melody_loading');
                }
                this.setState({ youtubeLoading: false });
            });
    };

    add = () => {

    }

    render() {
        return (
            <div>
                <Form
                    ref={this.formRef}
                    name="validate_other"
                    {...formItemLayout}
                    onFinish={this.onFinish}
                    initialValues={{
                        tonic: 'C',
                        meter: '4/4',
                        mode: 'maj',
                        tempo: 120,
                        chord_style: 'pop_standard',
                        rhythm_density:2,
                        voice_number:2,
                        enable_texture_style:true,
                        enable_chord_style:true,
                        use_vocal_only: true,
                        phrases: [{ phrase_name: 'A', phrase_length: 8 }],
                    }}
                >
                    <Divider orientation="left">Full song (YouTube)</Divider>
                    <Form.Item
                        label="YouTube URL"
                        name="youtube_url"
                        extra="Paste a YouTube or YouTube Music link, then Load melody. Toggle Vocal-only extraction to choose whether Demucs first isolates vocals before Basic Pitch. Vocal-only is cleaner but much slower; turning it off transcribes the full mix. FFmpeg is required for yt-dlp; first-time Demucs use may download large model weights (slow on CPU)."
                    >
                        <Input.Search
                            placeholder="https://www.youtube.com/watch?v=… or https://music.youtube.com/watch?v=…"
                            enterButton="Load melody"
                            loading={this.state.youtubeLoading}
                            onSearch={this.onYoutubeLoad}
                            allowClear
                        />
                    </Form.Item>
                    <Form.Item
                        name="use_vocal_only"
                        label="Vocal-only extraction (Demucs)"
                        valuePropName="checked"
                        extra="On: separate vocals first (Demucs), then transcribe vocals. Off: transcribe full mix (faster, less clean)."
                    >
                        <Checkbox />
                    </Form.Item>

                    <Divider orientation="left">Set Phrase</Divider>
                    <Form.Item
                        label="Phrases"
                        extra="After Load melody, phrases are filled automatically (edit if needed). You can add or remove rows."
                    >
                        <Form.List name="phrases">
                            {(fields, { add, remove }) => (
                                <>
                                    {fields.map((field) => (
                                        <Space key={field.key} align="baseline" size={120}>
                                            <Form.Item
                                                {...field}
                                                label="Phrase Name"
                                                name={[field.name, 'phrase_name']}
                                                fieldKey={[field.fieldKey, 'phrase_name']}
                                                rules={[{ required: true, message: 'Missing name' }]}
                                            >
                                                <Select placeholder="Please select a name">
                                                    <Option key="phrase_name_option_A" value="A">A<span style={{ visibility: 'hidden' }}>____________________</span></Option>
                                                    <Option key="phrase_name_option_B" value="B">B<span style={{ visibility: 'hidden' }}>____________________</span></Option>
                                                    <Option key="phrase_name_option_C" value="C">C<span style={{ visibility: 'hidden' }}>____________________</span></Option>
                                                    <Option key="phrase_name_option_D" value="D">D<span style={{ visibility: 'hidden' }}>____________________</span></Option>
                                                </Select>
                                            </Form.Item>
                                            <Form.Item
                                                {...field}
                                                label="Phrase Length"
                                                name={[field.name, 'phrase_length']}
                                                fieldKey={[field.fieldKey, 'phrase_length']}
                                                rules={[{ required: true, message: 'Missing length' }]}
                                            >
                                                <Select placeholder="Please select a length">
                                                    <Option key="phrase_length_option_4" value={4}>4<span style={{ visibility: 'hidden' }}>_____________________</span></Option>
                                                    <Option key="phrase_length_option_8" value={8}>8<span style={{ visibility: 'hidden' }}>_____________________</span></Option>
                                                </Select>
                                            </Form.Item>

                                            <MinusCircleOutlined onClick={() => remove(field.name)} />
                                        </Space>
                                    ))}

                                    <Form.Item>
                                        <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />}>
                                            Add Phrase
                                        </Button>
                                    </Form.Item>
                                </>
                            )}
                        </Form.List>
                    </Form.Item>

                    <Divider orientation="left">Set Meta</Divider>

                    <Form.Item name="tonic" label="Tonic" hasFeedback
                        rules={[{ required: true, message: 'Please select a tonic!' }]}
                    >
                        <Select placeholder="Please select a tonic">
                            {tonic.map((item) => {
                                return <Option key={`tonic_option_${item}`} value={item}>{item}</Option>
                            })}
                        </Select>
                    </Form.Item>

                    <Form.Item name="mode" label="Mode" hasFeedback
                        rules={[{ required: true, message: 'Please select a mode!' }]}
                    >
                        <Select placeholder="Please select a mode">
                            {mode.map((item) => {
                                return <Option key={`mode_option_${item}`} value={item.value}>{item.ui}</Option>
                            })}
                        </Select>
                    </Form.Item>

                    <Form.Item name="meter" label="Meter" hasFeedback
                        rules={[{ required: true, message: 'Please select a meter!' }]}
                    >
                        <Select placeholder="Please select a meter">
                            {meter.map((item) => {
                                return <Option key={`meter_option_${item}`} value={item.value} disabled={item.value === '3/4'?true:false}>{item.ui}</Option>
                            })}
                        </Select>
                    </Form.Item>

                    <Form.Item name="tempo" label="Tempo (BPM)"
                        rules={[{ required: true, message: 'Please set tempo!' }]}
                        extra="Auto-filled after Load melody. Adjust if the estimate sounds off."
                    >
                        <InputNumber min={30} max={260} step={1} style={{ width: '100%' }} />
                    </Form.Item>

                    <Divider orientation="left">Set Style</Divider>

                    <Form.Item name="enable_chord_style" label='Enable Chord Style Controlling' valuePropName="checked">
                        <Checkbox checked={this.state.chordStyleControl} onChange={(e) => this.setState({chordStyleControl:e.target.checked})}/>
                    </Form.Item>

                    <Form.Item name="chord_style" label="Chord Style"
                        rules={[{ required: true, message: 'Please select a style!' }]}
                    >
                        <Select placeholder="Please select a style" disabled={!this.state.chordStyleControl}>
                            {style.map((item) => {
                                return <Option key={`chord_style_option_${item}`} value={item.value}>{item.ui}</Option>
                            })}
                        </Select>
                    </Form.Item>

                    <Form.Item name="enable_texture_style" label='Enable Texture Style Controlling' valuePropName="checked">
                        <Checkbox checked={this.state.textureStyleControl} onChange={(e) => this.setState({textureStyleControl:e.target.checked})} />
                    </Form.Item>

                    <Form.Item name="rhythm_density" label="Texture Rhythm Density (RD)"
                        rules={[{ required: true, message: 'Please select a RD!' }]}
                    >
                        <Slider max={4} min={0} step={1} dots disabled={!this.state.textureStyleControl}/>
                    </Form.Item>

                    <Form.Item name="voice_number" label="Texture Voice Number (VN)"
                        rules={[{ required: true, message: 'Please select a VN!' }]}
                    >
                        <Slider max={4} min={0} step={1} dots disabled={!this.state.textureStyleControl}/>
                    </Form.Item>

                    <Form.Item
                        wrapperCol={{
                            span: 4,
                            offset: 11,
                        }}
                        style={{ marginTop: '60px' }}
                    >
                        <Button type="primary" htmlType="submit" shape='round'>
                            Begin Generate
                        </Button>
                    </Form.Item>
                </Form>
                <div style={{ minHeight: '100px' }}></div>
            </div>
        )
    }
}
