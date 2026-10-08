//! The brain hub: one shared reranker per data dir, so N sessions do not each
//! hold a 1.7 GB model. `serve` is the server; `status` and `stop` are the
//! operator's overrides. Unix only.

pub mod client;
pub mod endpoint;
pub mod proto;
pub mod server;

use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};

use self::proto::{Reply, Request};
use crate::config::Paths;

/// Why a rerank got no answer from the hub, as the search outcome and the
/// doctor's `hub answers` row name it. One spelling, used everywhere.
pub mod reason {
    pub const BUSY: &str = "hub-busy";
    pub const DOWN: &str = "hub-down";
    pub const STARTING: &str = "hub-starting";
    pub const UNSAFE: &str = "hub-unsafe";
    pub const OLDER: &str = "hub-older";
    /// What every one of them starts with.
    pub const PREFIX: &str = "hub-";
}

#[derive(clap::Subcommand)]
pub enum HubAction {
    /// Run the hub in the foreground. A brain process starts this by itself.
    Serve,
    /// Say whether a hub is running, and what it holds.
    Status,
    /// Ask the running hub to finish its work and exit.
    Stop,
}

/// # Errors
/// Whatever the action fails with; a hub that is simply not running is not
/// an error for `status` or `stop`.
pub fn run(action: HubAction) -> Result<()> {
    match action {
        HubAction::Serve => server::serve(),
        HubAction::Status => status(),
        HubAction::Stop => stop(),
    }
}

/// Connect the way a client does (see [`client::probe`]): the directory and
/// socket checked before a byte is sent, the lock held, and the pid in
/// `Welcome` the pid in the lock. `None` when no hub is running.
fn connect() -> Result<Option<client::Conn>> {
    let paths = Paths::resolve()?;
    let Ok(ep) = endpoint::resolve(&paths.data_dir) else { return Ok(None) };
    match client::probe(&ep, Duration::from_secs(3), Instant::now() + Duration::from_secs(6), None) {
        client::Probe::Absent { .. } => Ok(None),
        client::Probe::Unsafe(reason) => bail!("{}: {reason}", reason::UNSAFE),
        client::Probe::Down => bail!("cannot talk to the hub"),
        client::Probe::Hub(conn) if conn.usable => Ok(Some(conn)),
        client::Probe::Hub(_) => bail!("the hub refused: {}", reason::OLDER),
    }
}

/// `text` from the wire with every control character dropped, so a hub cannot
/// move the operator's cursor or rewrite their screen through `status`.
pub fn printable(text: &str) -> String {
    text.chars().filter(|c| !c.is_control()).collect()
}

fn status() -> Result<()> {
    let Some(mut conn) = connect()? else {
        println!("hub: not running");
        return Ok(());
    };
    match client::exchange(&mut conn, &Request::Status).map_err(|e| anyhow!(e))? {
        Reply::Status { pid, build, proto, uptime_ms, footprint_kb, clients, sessions, queued, model_loaded, retiring } => {
            println!("hub: running{}", if retiring { " (retiring)" } else { "" });
            println!("pid        {pid}");
            println!("build      {} (proto {proto})", printable(&build));
            println!("uptime     {}s", uptime_ms / 1000);
            match footprint_kb {
                Some(kb) => println!("footprint  {} MB", kb / 1024),
                None => println!("footprint  unknown"),
            }
            println!("clients    {clients}");
            println!("sessions   {sessions} in 10 min");
            println!("queued     {queued}");
            println!("model      {}", if model_loaded { "loaded" } else { "not loaded" });
            Ok(())
        }
        _ => bail!("the hub sent something unexpected"),
    }
}

fn stop() -> Result<()> {
    let Some(mut conn) = connect()? else {
        println!("hub: not running");
        return Ok(());
    };
    match client::exchange(&mut conn, &Request::Stop).map_err(|e| anyhow!(e))? {
        Reply::Ack => {
            println!("hub: stopping");
            Ok(())
        }
        _ => bail!("the hub sent something unexpected"),
    }
}

#[cfg(test)]
mod tests {
    use super::printable;

    #[test]
    fn status_text_from_the_wire_loses_its_control_characters() {
        assert_eq!(printable("0.68.0"), "0.68.0");
        assert_eq!(printable("0.1\x1b[2J\r\n\u{7}x"), "0.1[2Jx");
    }
}
