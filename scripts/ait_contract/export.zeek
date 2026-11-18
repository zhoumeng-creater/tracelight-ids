@load base/protocols/conn
@load base/protocols/http
@load base/protocols/ssl
@load base/protocols/dns
@load policy/tuning/json-logs
@load policy/frameworks/files/hash-all-files

event file_new(f: fa_file)
    { Files::add_analyzer(f, Files::ANALYZER_SHA256); }

# These are offline export clocks, never claimed to be detector ingestion times.
redef record Conn::Info += {
    contract_export_network_time: time &log &optional;
    contract_after_input_eof: bool &log &default=F;
    contract_during_shutdown: bool &log &default=F;
};

global contract_eof = F;
global contract_shutdown = F;

event Pcap::file_done(path: string) &priority=100
    { contract_eof = T; }

event net_done(t: time) &priority=100
    { contract_shutdown = T; }

# Base Conn writes at priority -5; set the extension after set_conn at +5.
event connection_state_remove(c: connection) &priority=0
    {
    c$conn$contract_export_network_time = network_time();
    c$conn$contract_after_input_eof = contract_eof;
    c$conn$contract_during_shutdown = contract_shutdown;
    }
