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
local function normalize(lang)
    lang = (lang or ''):lower():match('^[^_-]+') or ''
    local aliases = {zh='zh',zho='zh',chi='zh',chs='zh',cht='zh',cmn='zh',yue='zh',
                     en='en',eng='en',ja='ja',jpn='ja'}
    return aliases[lang] or lang
end
local function is_chinese(t)
    if normalize(t.lang) == 'zh' then return true end
    local title = t.title or ''
    for _, hint in ipairs({'中文','汉语','漢語','简体','簡體','繁体','繁體','中英','英中','中日','日中'}) do
        if title:find(hint, 1, true) then return true end
    end
    local name = (t['external-filename'] or ''):match('[^/\\]+$') or ''
    for token in name:lower():gmatch('[%a]+') do
        if normalize(token) == 'zh' then return true end
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
