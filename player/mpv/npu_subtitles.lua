-- Native mpv frontend. No daemon or process watcher; models live in a helper.
local mp = require 'mp'
local utils = require 'mp.utils'
local options = require 'mp.options'
local o = {root='', python='', enabled=true, ffmpeg='', ffprobe=''}
options.read_options(o, 'npu_subtitles')
local session, child, timer = 0, nil, nil
local loaded, manual, expected_sid = false, false, nil
local fingerprint, status = '', 'idle'
local seek_serial = 0
local last_overlay, last_aid = '', nil
local overlay = mp.create_osd_overlay('ass-events')
-- Must equal npu_subtitles.ass.OVERLAY_RES_X/Y used by the helper's events.
overlay.res_x, overlay.res_y, overlay.z = 1280, 720, 0

local function state()
    mp.set_property_native('user-data/npu-subtitles', {
        token=tostring(session), status=status, manual=manual,
        active=loaded and o.enabled and not manual,
        seek=seek_serial,
    })
end
local function clear()
    last_overlay = ''
    overlay:remove()
end
local function stop()
    session = session + 1
    clear()
    if child then mp.abort_async_command(child); child = nil end
    if loaded and o.enabled and not manual then status='switching' end
    state()
end
-- Mirrors npu_subtitles.routing (ALIASES, CHINESE_TITLE_HINTS, TITLE_HINTS,
-- SUFFIX_TAGS and track_language); tests compare these tables with Python.
local aliases = {
    ['zh']='zh', ['zho']='zh', ['chi']='zh', ['chs']='zh', ['cht']='zh', ['cmn']='zh',
    ['yue']='zh', ['chinese']='zh', ['mandarin']='zh', ['en']='en', ['eng']='en', ['english']='en',
    ['ja']='ja', ['jpn']='ja', ['japanese']='ja', ['ko']='ko', ['kor']='ko', ['korean']='ko',
    ['한국어']='ko', ['es']='es', ['spa']='es', ['spanish']='es', ['español']='es', ['fr']='fr',
    ['fra']='fr', ['fre']='fr', ['french']='fr', ['français']='fr', ['de']='de', ['deu']='de',
    ['ger']='de', ['german']='de', ['deutsch']='de', ['pt']='pt', ['por']='pt', ['portuguese']='pt',
    ['português']='pt', ['it']='it', ['ita']='it', ['italian']='it', ['italiano']='it', ['ru']='ru',
    ['rus']='ru', ['russian']='ru', ['русский']='ru', ['hi']='hi', ['hin']='hi', ['hindi']='hi',
    ['हिन्दी']='hi', ['हिंदी']='hi',
}
local chinese_title_hints = {
    ['中文']=true, ['汉语']=true, ['漢語']=true, ['简体']=true, ['簡體']=true, ['繁体']=true,
    ['繁體']=true, ['中英']=true, ['英中']=true, ['日中']=true, ['中日']=true,
}
local title_hints = {
    ['日本語']='ja', ['日语']='ja', ['日語']='ja', ['韩语']='ko', ['韓語']='ko',
    ['西班牙语']='es', ['西班牙語']='es', ['法语']='fr', ['法語']='fr', ['德语']='de',
    ['德語']='de', ['葡萄牙语']='pt', ['葡萄牙語']='pt', ['意大利语']='it', ['義大利語']='it',
    ['俄语']='ru', ['俄語']='ru', ['印地语']='hi', ['印地語']='hi',
}
local suffix_tags = {
    ['']=true, ['br']=true, ['cc']=true, ['cn']=true, ['default']=true, ['forced']=true,
    ['gb']=true, ['hans']=true, ['hant']=true, ['sdh']=true, ['tw']=true, ['us']=true,
}
local function contains_any(text, hints)
    for hint in pairs(hints) do
        if text:find(hint, 1, true) then return true end
    end
    return false
end
-- Like Python re.split on [\s.\[\]()_-]+, keeping empty leading/trailing parts.
local function split(text)
    local parts, start = {}, 1
    while true do
        local first, last = text:find('[%s%.%[%]%(%)_%-]+', start)
        if not first then parts[#parts+1] = text:sub(start); return parts end
        parts[#parts+1] = text:sub(start, first - 1)
        start = last + 1
    end
end
local function language(value)
    local raw = (value or ''):match('^%s*(.-)%s*$'):lower():gsub('_', '-')
    local code = aliases[raw:match('^[^-]*')]
    if code then return code end
    if raw == 'und' or raw == 'unknown' then return '' end
    return raw
end
local function is_chinese(t)
    local title = t.title or ''
    if contains_any(title, chinese_title_hints) then return true end
    local declared = language(t.lang)
    if declared ~= '' then return declared == 'zh' end
    if contains_any(title, title_hints) then return false end
    for _, token in ipairs(split(title:lower())) do
        if aliases[token] then return aliases[token] == 'zh' end
    end
    -- A language suffix, not a language word in the movie's own title.
    local name = (t['external-filename'] or ''):match('[^/\\]*$')
    local tokens = split((name:match('^(.*)%.') or name):lower())
    for i = #tokens, 2, -1 do
        if aliases[tokens[i]] then return aliases[tokens[i]] == 'zh' end
        if not suffix_tags[tokens[i]] then break end
    end
    return false
end
local function select_sub(sid)
    expected_sid = tostring(sid)
    mp.set_property('sid', expected_sid)
    mp.set_property('secondary-sid', 'no')
    mp.set_property_bool('sub-visibility', true)
end
local function route()
    timer = nil
    if not loaded or not o.enabled or manual then return end
    local tracks = mp.get_property_native('track-list', {})
    local parts = {mp.get_property('path', ''), mp.get_property('aid', 'no')}
    local chinese = nil
    for _, t in ipairs(tracks) do
        if t.type == 'sub' then
            parts[#parts+1] = utils.format_json({t.id,t.lang or '',t.title or '',t.codec or '',t['external-filename'] or ''})
            if is_chinese(t) and (not chinese or (t.default and not chinese.default)) then chinese = t end
        end
    end
    local key = table.concat(parts, '|')
    if key == fingerprint then return end
    fingerprint = key
    stop()
    if chinese then
        select_sub(chinese.id)
        status = 'native-zh'; state(); return
    end
    local path = mp.get_property('path', '')
    if path == '' or path:match('^%a[%w+.-]*://') then status='unsupported-source'; state(); return end
    if not path:match('^%a:[/\\]') and not path:match('^[/\\]') then
        path = utils.join_path(mp.get_property('working-directory'), path)
    end
    if o.root == '' then status='not-configured'; state(); mp.msg.warn('Set root in script-opts/npu_subtitles.conf'); return end
    local python = o.python ~= '' and o.python or utils.join_path(o.root, '.venv/Scripts/pythonw.exe')
    local endpoint = mp.get_property('input-ipc-server', '')
    if endpoint == '' then
        endpoint = '\\\\.\\pipe\\npu-subtitles-' .. tostring(utils.getpid())
        mp.set_property('input-ipc-server', endpoint)
    end
    select_sub('no')
    local token = tostring(session)
    local args = {python, utils.join_path(o.root, 'mpv_plugin.py'), '--ipc', endpoint,
                  '--token', token, '--video', path, '--pid', tostring(utils.getpid())}
    if o.ffmpeg ~= '' then args[#args+1]='--ffmpeg'; args[#args+1]=o.ffmpeg end
    if o.ffprobe ~= '' then args[#args+1]='--ffprobe'; args[#args+1]=o.ffprobe end
    status='starting'; state()
    child = mp.command_native_async({name='subprocess', args=args, playback_only=false,
        capture_stdout=false, capture_stderr=false}, function(success, result, error)
        if token ~= tostring(session) then return end
        child=nil
        if not success or (result and result.status ~= 0) then
            clear(); status='error'; state()
            mp.msg.error('NPU subtitle helper failed; see project outputs/plugin logs. ' .. tostring(error or ''))
        end
    end)
end
local function schedule()
    if timer then timer:kill() end
    timer = mp.add_timeout(0.15, route)
end
mp.register_event('start-file', function()
    loaded=false; manual=false; expected_sid=nil; fingerprint=''; status='loading'; stop()
    -- Only exact-basename sidecars; unrelated files must not win Chinese priority.
    mp.set_property('file-local-options/sub-auto', 'exact')
end)
mp.register_event('file-loaded', function() loaded=true; schedule() end)
mp.register_event('end-file', function() loaded=false; status='idle'; stop() end)
mp.register_event('shutdown', stop)
mp.observe_property('track-list', 'native', schedule)
mp.observe_property('aid', 'string', function(_, value)
    if loaded and last_aid and value ~= last_aid then stop(); fingerprint='' end
    last_aid=value
    if loaded then schedule() end
end)
mp.observe_property('sid', 'string', function(_, value)
    if loaded and expected_sid and value ~= expected_sid then
        manual=true; status='manual'; stop()
    end
end)
mp.register_event('seek', function() seek_serial=seek_serial+1; clear(); state() end)
mp.observe_property('sub-visibility', 'bool', function(_, value)
    if not value then overlay:remove()
    elseif last_overlay ~= '' then overlay.data=last_overlay; overlay:update() end
end)
mp.register_script_message('npu-status', function(token, value)
    if token == tostring(session) then status=value; state() end
end)
mp.register_script_message('npu-select', function(token, sid)
    if token == tostring(session) then select_sub(sid) end
end)
mp.register_script_message('npu-overlay', function(token, seek, text)
    if token ~= tostring(session) or seek ~= tostring(seek_serial) or not loaded or manual then return end
    last_overlay=text
    if text == '' or not mp.get_property_bool('sub-visibility', true) then overlay:remove(); return end
    overlay.data = text; overlay:update()
end)
mp.add_key_binding('Ctrl+n', 'toggle-npu-subtitles', function()
    if manual or not o.enabled then
        manual=false; o.enabled=true; fingerprint=''; schedule()
    else
        o.enabled=false; status='disabled'; stop(); expected_sid=nil
    end
    mp.osd_message(o.enabled and 'NPU subtitles: auto' or 'NPU subtitles: off', 2)
end)
state()
