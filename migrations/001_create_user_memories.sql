-- ═══════════════════════════════════════════════════════════════════════════
-- Migration 001 — Table public.user_memories
-- À exécuter dans le SQL Editor de Supabase.
--
-- Contexte : core/user_memory_service.py lit/écrit cette table pour la
-- mémoire personnelle par utilisateur (source du PGRST205 tant qu'elle
-- n'existe pas). Le schéma correspond exactement aux DTO de
-- models/user_memory.py (UserMemoryBase / UserMemory).
-- ═══════════════════════════════════════════════════════════════════════════

create extension if not exists "pgcrypto";

create table if not exists public.user_memories (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null references auth.users (id) on delete cascade,
  content    text not null check (length(btrim(content)) > 0),
  source     text not null default 'manual' check (length(btrim(source)) > 0),
  metadata   jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists user_memories_user_id_idx
  on public.user_memories (user_id);

-- updated_at automatique
create or replace function public.user_memories_set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists user_memories_updated_at on public.user_memories;
create trigger user_memories_updated_at
  before update on public.user_memories
  for each row execute function public.user_memories_set_updated_at();

-- ── Row Level Security : chaque utilisateur ne voit QUE ses mémoires ────────
alter table public.user_memories enable row level security;

drop policy if exists "user_memories_select_own" on public.user_memories;
create policy "user_memories_select_own" on public.user_memories
  for select to authenticated
  using (auth.uid() = user_id);

drop policy if exists "user_memories_insert_own" on public.user_memories;
create policy "user_memories_insert_own" on public.user_memories
  for insert to authenticated
  with check (auth.uid() = user_id);

drop policy if exists "user_memories_update_own" on public.user_memories;
create policy "user_memories_update_own" on public.user_memories
  for update to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists "user_memories_delete_own" on public.user_memories;
create policy "user_memories_delete_own" on public.user_memories
  for delete to authenticated
  using (auth.uid() = user_id);

-- ── Grants ──────────────────────────────────────────────────────────────────
-- service_role : accès complet (le backend filtre lui-même par user_id).
-- authenticated : accès via RLS uniquement.
grant all on table public.user_memories to service_role;
grant select, insert, update, delete on table public.user_memories to authenticated;
