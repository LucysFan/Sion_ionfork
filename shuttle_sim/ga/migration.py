"""Migration topologies for island GA."""
import numpy as np

class RingMigration:
    """Directed ring S→A→V→R→H→S, every `interval` gens, `n_migrants` per island.
    Поддерживает изолированные острова: если island.isolated_until > generation,
    остров не участвует в миграции (эволюционирует сам), а потом
    в момент де-изоляции делает burst в общую систему.
    """
    def __init__(self, islands, interval=5, n_migrants=2):
        self.islands=islands
        self.interval=interval
        self.n_migrants=n_migrants
        self.history=[]
        self.stats={"total":0,"active":0,"isolated_intra":0,"bursts":0,"global_shuffles":0}

    def _is_isolated(self, isl, generation):
        return getattr(isl, 'isolated_until', 0) > generation

    def should_migrate(self, generation):
        if generation==0:
            return False
        # de-isolation burst always migrates even off-interval
        deisolated = [isl for isl in self.islands if getattr(isl,'isolated_until',0)==generation]
        if deisolated:
            return True
        if generation % self.interval !=0:
            # дополнительно проверяем intra-isolated каждые 2*interval
            if generation % (self.interval*2) ==0:
                isolated = [isl for isl in self.islands if self._is_isolated(isl, generation)]
                if len(isolated) >= 2:
                    return True
            return False
        active = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
        isolated = [isl for isl in self.islands if self._is_isolated(isl, generation)]
        return len(active) >= 2 or len(isolated) >= 2

    def migrate(self, generation):
        # burst from just-deisolated islands into active pool — selective: only good islands burst
        deisolated = [isl for isl in self.islands if getattr(isl,'isolated_until',0)==generation]
        if deisolated:
            # селективная миграция: только от хороших островов (не ломает популяцию)
            # считаем медиану активных best
            active = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
            active_bests = []
            for isl in active:
                try:
                    fits=[g.fitness for g in isl.population if g.fitness is not None and g.fitness<1e6]
                    if fits: active_bests.append(min(fits))
                except: pass
            median_active = float(np.median(active_bests)) if active_bests else 1.0
            # также глобальный лучший
            global_best = min(active_bests) if active_bests else 1.0
            for src in deisolated:
                # оценим src best
                try:
                    src_fits=[g.fitness for g in src.population if g.fitness is not None and g.fitness<1e6]
                    src_best = min(src_fits) if src_fits else 1e6
                except:
                    src_best=1e6
                # пропускаем плохие источники
                # критерии: 1) src_best должен быть < 1e5 (не rejected), 2) < median*2 или < global_best*10, 3) если активных нет — < 100
                skip=False
                # R-остров несёт робастность, но его фитнес естественно хуже (primary+robust) ≈ 5-8 vs медиана 0.5 —
                # даём ему послабление ×1.8, иначе он никогда не прорвётся через selective burst.
                is_robust_src = str(getattr(src,'strategy','')).startswith('R')
                median_mult = 5.0 if is_robust_src else 4.0
                global_mult = 12.0 if is_robust_src else 10.0
                if src_best >= 1e5:
                    skip=True
                elif active_bests:
                    # если src хуже медианы в median_mult раз И хуже глобального в global_mult — скип
                    if src_best > median_active*median_mult and src_best > global_best*global_mult:
                        skip=True
                    # также если src_best > 1.0 и медиана <0.05 (плохой остров среди хороших) — скип (было 0.5 слишком строго)
                    if median_active < 0.05 and src_best > 1.0:
                        skip=True
                else:
                    if src_best > 100:
                        skip=True
                if skip:
                    self.history.append(dict(generation=generation, burst_from=src.id, isolated_until=src.isolated_until, type="burst_skipped", src=src.id, reason=f"bad fitness {src_best:.3g} vs median {median_active:.3g}", dst=[]))
                    self.stats["total"]+=1
                    continue
                migrants = src.get_migrants(self.n_migrants, diverse=True)
                # дополнительно фильтруем мигрантов: только если fitness мигранта < worst dst *0.95
                for dst in self.islands:
                    if dst is src or self._is_isolated(dst, generation):
                        continue
                    # dst worst
                    try:
                        dst_fits=[g.fitness for g in dst.population if g.fitness is not None and g.fitness<1e6]
                        dst_worst = max(dst_fits) if dst_fits else 1e6
                    except:
                        dst_worst=1e6
                    # если мигрант хуже чем worst dst — не заменяем (чтобы не ломать)
                    mig_fit = migrants[0].fitness if migrants and migrants[0].fitness is not None else 1e6
                    if mig_fit is not None and mig_fit < dst_worst*0.95:
                        dst.replace_worst(migrants[:1])
                    else:
                        # мягкая замена: не трогаем если мигрант плохой
                        pass
                self.history.append(dict(generation=generation, burst_from=src.id, isolated_until=src.isolated_until, type="burst", src=src.id, dst=[d.id for d in self.islands if d is not src and not self._is_isolated(d,generation)], src_best=float(src_best), median_active=float(median_active)))
                self.stats["bursts"]+=1
                self.stats["total"]+=1
        active = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
        isolated = [isl for isl in self.islands if self._is_isolated(isl, generation)]
        # active ring
        if len(active) >= 2 and generation % self.interval ==0:
            migrants_per_island={}
            for isl in active:
                migrants_per_island[isl.id]=isl.get_migrants(self.n_migrants, diverse=True)
            n=len(active)
            edges=[]
            for i, isl in enumerate(active):
                src_id=active[(i-1)%n].id
                incoming=migrants_per_island[src_id]
                isl.replace_worst(incoming)
                edges.append(dict(src=src_id, dst=isl.id, n=len(incoming)))
            self.history.append(dict(generation=generation, migrant_ids=list(migrants_per_island.keys()),
                                     isolated_skipped=[isl.id for isl in isolated],
                                     type="ring_active", edges=edges))
            self.stats["active"]+=1
            self.stats["total"]+=1
            if generation % (self.interval*3)==0:
                self.global_shuffle(generation)
        # intra-isolated ring (sometimes)
        if len(isolated) >= 2 and generation % (self.interval*2)==0:
            # с вероятностью 70% мигрируем внутри изоляции, чтобы показать жизнь в зонах карантина
            migrants_per_island={}
            for isl in isolated:
                migrants_per_island[isl.id]=isl.get_migrants(max(1,self.n_migrants//2), diverse=True)
            n=len(isolated)
            edges=[]
            for i, isl in enumerate(isolated):
                src_id=isolated[(i-1)%n].id
                incoming=migrants_per_island[src_id]
                isl.replace_worst(incoming)
                edges.append(dict(src=src_id, dst=isl.id, n=len(incoming)))
            self.history.append(dict(generation=generation, migrant_ids=list(migrants_per_island.keys()),
                                     type="ring_isolated", edges=edges, isolated_group=[isl.id for isl in isolated]))
            self.stats["isolated_intra"]+=1
            self.stats["total"]+=1
        # если оба кольца не сработали, но should_migrate вернул True из-за deisolated, уже записали burst
        if generation % (self.interval*3)==0 and len(active)>=2:
            # global shuffle уже вызван выше при active; если isolated тоже, не дублируем
            pass

    def global_shuffle(self, generation=None):
        pool_islands = self.islands
        if generation is not None:
            pool_islands = [isl for isl in self.islands if not self._is_isolated(isl, generation)]
        if len(pool_islands) < 2:
            return
        pool=[isl.rng.choice(isl.population).copy() for isl in pool_islands]
        for isl in pool_islands:
            idx=isl.rng.integers(2, len(isl.population))
            donor=isl.rng.choice(pool).copy()
            adapted=__import__('shuttle_sim.ga.genome', fromlist=['Genome']).Genome(donor.theta, isl.strategy)
            adapted.fitness=None
            isl.population[idx]=adapted
        try:
            self.stats["global_shuffles"]+=1
            self.history.append(dict(generation=generation, type="global_shuffle", islands=[isl.id for isl in pool_islands]))
        except: pass

class StarMigration(RingMigration):
    """Star: all islands send to hub (R robust) and hub broadcasts."""
    def migrate(self, generation):
        hub=self.islands[3] if len(self.islands)>3 else self.islands[0]
        # collect from all
        all_migrants=[]
        for isl in self.islands:
            if isl is hub: continue
            all_migrants.extend(isl.get_migrants(1, diverse=False))
        # hub replaces worst with each
        for m in all_migrants:
            hub.replace_worst([m])
        # hub broadcasts its best to all
        bests=hub.get_migrants(1, diverse=False)
        for isl in self.islands:
            if isl is hub: continue
            isl.replace_worst(bests)
        self.history.append(dict(generation=generation, star=True))
